import re
from io import BytesIO

import fitz
from docx import Document
from nltk import sent_tokenize

from src.core.settings import logger


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
            if filename.endswith(".txt"):
                return await self._extract_text_from_txt(file_bytes)
            elif filename.endswith(".docx"):
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

    async def _extract_text_from_pdf(self, file_bytes: BytesIO) -> tuple[str, bool] | tuple[None, bool]:
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
            result = " ".join([page.get_text() for page in doc.pages()])

            logger.info(f"Text extracted successfully")
            cleaned_text = await self.clean_text(result)
            return cleaned_text, True
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
