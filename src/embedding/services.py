import asyncio
import re
import uuid
from io import BytesIO
from typing import Any

import fitz
from docx import Document
from fastapi import UploadFile
from nltk import sent_tokenize
from openai import AzureOpenAI
from openai.types import CreateEmbeddingResponse

from src.core.settings import logger, settings
from src.embedding.vector_db import add_embedding


class TextExtractorService:
    async def extract_text(self, filename: str, file_bytes: BytesIO) -> tuple[str, bool] | tuple[None, bool]:
        """
        Extracts text from a file bytes steam based on the file extension.

        :param filename: File name.
        :param file_bytes: File content as a BytesIO object.
        :return:
            - Tuple (`str`, `True`) if the text is extracted successfully.
            - Tuple (`None`, `False`) if an error occurs or unsupported file type.
        """

        try:
            # if filename.endswith(".txt"):
            #     return await self._extract_text_from_txt(file_bytes)
            if filename.endswith(".docx"):
                return await self._extract_text_from_docx(file_bytes)
            elif filename.endswith(".pdf"):
                return await self._extract_text_from_pdf(file_bytes)
        except Exception as e:
            logger.error(f"TextExtractorService {str(e)}")
            return None, False

        return "Unsupported file type", False

    async def _extract_text_from_txt(self, file_bytes: BytesIO) -> tuple[str, bool] | tuple[None, bool]:
        """
        Extracts text from a text file.

        :param file_bytes: File content as a BytesIO object.
        :return:
            - Tuple (`str`, `True`) if the text is extracted successfully.
            - Tuple (`None`, `False`) if an error occurs.
        """

        try:
            logger.info(f"Extracting text from TXT file")

            result = file_bytes.read().decode("utf-8")
            cleaned_text = await self.clean_text(result)

            return cleaned_text, True
        except Exception as e:
            logger.error(f"TextExtractorService {str(e)}")
            return None, False

    async def _extract_text_from_docx(self, file_bytes: BytesIO) -> tuple[str, bool] | tuple[None, bool]:
        """
        Extracts text from a DOCX file bytes using the `python-docx` library.

        :param file_bytes:  File content as a BytesIO object.
        :return:
            - Tuple (`str`, `True`) if the text is extracted successfully.
            - Tuple (`None`, `False`) if an error occurs.
        """

        try:
            logger.info(f"Starting text extraction from DOCX file")
            doc = Document(file_bytes)
            result = " ".join([para.text for para in doc.paragraphs if para.text.strip()])
            logger.info(f"Text extracted successfully")

            cleaned_text = await self.clean_text(result)
            return cleaned_text, True
        except Exception as e:
            logger.error(f"TextExtractorService {str(e)}")
            return None, False

    async def _extract_text_from_pdf(self, file_bytes: BytesIO) -> tuple[dict[int, str], bool] | tuple[None, bool]:
        """
        Extracts text from a PDF file bytes using the `PyMuPDF` library.

        :param file_bytes: File content as a BytesIO object.
        :return:
            - Tuple (`str`, `True`) if the text is extracted successfully.
            - Tuple (`None`, `False`) if an error occurs
        """

        try:
            logger.info(f"Starting text extraction from PDF file")
            doc = fitz.open("pdf", file_bytes.read())

            page_texts = {}
            for page in doc.pages():
                text = page.get_text()
                cleaned_text = await self.clean_text(text)
                if cleaned_text.strip():
                    page_texts[page.number + 1] = cleaned_text

            logger.info(f"Text extracted successfully")
            # cleaned_text = await self.clean_text(result)
            return page_texts, True
        except Exception as e:
            logger.error(f"TextExtractorService {str(e)}")
            return None, False

    @staticmethod
    async def clean_text(text) -> str:
        cleaned = re.sub(r"[\n\r\t\b]", " ", text)
        cleaned = re.sub(r"\s+", " ", cleaned)
        cleaned = cleaned.strip()

        logger.info(f"Cleaning text from {len(cleaned)} characters")
        return cleaned


class SentenceAwareChunker:
    def __init__(self, tokenizer, max_tokens: int = 500):
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens

    async def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    async def chunk_text(self, text: str) -> list[str]:
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = ""
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = await self.count_tokens(sentence)

            if current_tokens + sentence_tokens <= self.max_tokens:
                current_chunk += " " + sentence.strip()
                current_tokens += sentence_tokens
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())

                if sentence_tokens > self.max_tokens:
                    token_ids = self.tokenizer.encode(sentence)
                    for i in range(0, len(token_ids), self.max_tokens):
                        chunk = self.tokenizer.decode(token_ids[i : i + self.max_tokens])
                        chunks.append(chunk.strip())
                    current_chunk = ""
                    current_tokens = 0
                else:
                    current_chunk = sentence.strip()
                    current_tokens = sentence_tokens

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks


class CreateEmbeddingService(SentenceAwareChunker):
    def __init__(
        self, embedding_client: AzureOpenAI, text_extractor: TextExtractorService, tokenizer, max_tokens: int = 500
    ):
        super().__init__(tokenizer, max_tokens)

        self.embedding_client = embedding_client
        self.text_extractor = text_extractor
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens

    async def create_embeddings(self, user_id: str, text: str = None, file: UploadFile = None) -> dict[str, Any]:
        text_chunks = []

        if text:
            chunks = await self.chunk_text(text)
            text_chunks.extend([{"text": c} for c in chunks])

        if file:
            file_bytes = await file.read()
            file_stream = BytesIO(file_bytes)
            file_chunks, is_extracted = await self._extract_text_as_chunks(file.filename, file_stream)

            if not is_extracted:
                logger.error(f"Failed to extract text from file: {file.filename}")
                return {"status": "error", "message": "Failed to extract text from file."}

            text_chunks.extend(file_chunks)

        if not text_chunks:
            return {"status": "error", "message": "No text provided."}

        cleaned_chunks = await self._clean_text_chunks(text_chunks)
        model_response = await self.send_chunks_to_embedding_service(cleaned_chunks)
        if not model_response or not model_response.data:
            logger.error("Failed to create embeddings.")
            return {"status": "error", "message": "Failed to create embeddings."}

        await self._add_chunks_to_vector_db(text_chunks, model_response, user_id)

        return {"status": "success", "chunks_saved": model_response}

    async def send_chunks_to_embedding_service(self, text_chunks: list[str]) -> CreateEmbeddingResponse | None:
        try:
            response = await asyncio.to_thread(
                self.embedding_client.embeddings.create,
                input=text_chunks,
                model=settings.AZURE_OPENAI_DEPLOYMENT_NAME,
            )

            return response
        except Exception as e:
            logger.error(f"Error sending chunks to embedding service: {str(e)}")
            return None

    async def _extract_text_as_chunks(self, filename: str, file_stream: BytesIO) -> tuple[list[dict] | None, bool]:
        extracted_parts, is_extracted = await self.text_extractor.extract_text(filename, file_stream)
        if not is_extracted:
            logger.error(f"Failed to extract text from file: {filename}")
            return None, False

        text_chunks = []
        for part_number, part_text in extracted_parts.items():
            chunks = await self.chunk_text(part_text)
            for chunk in chunks:
                text_chunks.append({"text": chunk, "part": part_number})

        return text_chunks, True

    async def _add_chunks_to_vector_db(self, text_chunks: list[dict], model_response, user_id: str) -> None:
        for chunk_data, embedding_data in zip(text_chunks, model_response.data):
            embedding = embedding_data.embedding
            point_id = str(uuid.uuid4())

            payload = {"id": point_id, "user_id": user_id, "text": chunk_data["text"]}

            if "part" in chunk_data:
                payload["part"] = chunk_data["part"]

            await add_embedding(vector=embedding, payload=payload)

    async def _clean_text_chunks(self, text_chunks: list[dict]) -> list[str]:
        cleaned_texts = []
        for chunk in text_chunks:
            cleaned_text = await self.text_extractor.clean_text(chunk["text"])
            cleaned_texts.append(cleaned_text)

        return cleaned_texts


async def get_text_extractor_service() -> TextExtractorService:
    """:return: TextExtractorService instance."""
    return TextExtractorService()


async def get_text_tokenization_service(tokenizer, max_tokens: int = 500) -> SentenceAwareChunker:
    """
    :param tokenizer: Tokenizer instance.
    :param max_tokens: Maximum number of tokens per chunk.
    :return: SentenceAwareChunker instance.
    """
    return SentenceAwareChunker(tokenizer, max_tokens)


async def get_embedding_service(
    embedding_client, text_extractor: TextExtractorService, tokenizer, max_tokens: int = 500
) -> CreateEmbeddingService:
    """
    :param embedding_client: Client for creating embeddings.
    :param text_extractor: TextExtractorService instance.
    :param tokenizer: Tokenizer instance.
    :param max_tokens: Maximum number of tokens per chunk.
    :return: CreateEmbeddingService instance.
    """
    return CreateEmbeddingService(embedding_client, text_extractor, tokenizer, max_tokens)
