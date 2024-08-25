from typing import Optional, Dict, Any, List, Set, Tuple, Union, IO
from pydantic import BaseModel, ConfigDict
from pathlib import Path
from src.utils import logger
from src.embed import Embedder
from urllib.parse import urljoin, urlparse
import time
import random
import httpx

try:
    from bs4 import BeautifulSoup  # noqa: F401
except ImportError:
    raise ImportError("The `bs4` package is not installed. Please install it via `pip install beautifulsoup4`.")

class Document(BaseModel):
    """Model for managing a document"""

    content: str
    id: Optional[str] = None
    name: Optional[str] = None
    meta_data: Dict[str, Any] = {}
    embedder: Optional[Embedder] = None
    embedding: Optional[List[float]] = None
    usage: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def embed(self, embedder: Optional[Embedder] = None) -> None:
        """Embed the document using the provided embedder"""

        _embedder = embedder or self.embedder
        if _embedder is None:
            raise ValueError("No embedder provided")

        self.embedding, self.usage = _embedder.get_embedding_and_usage(self.content)

    def to_dict(self) -> Dict[str, Any]:
        """Returns a dictionary representation of the document"""

        return self.model_dump(include={"name", "meta_data", "content"}, exclude_none=True)

    @classmethod
    def from_dict(cls, document: Dict[str, Any]) -> "Document":
        """Returns a Document object from a dictionary representation"""

        return cls.model_validate(**document)

    @classmethod
    def from_json(cls, document: str) -> "Document":
        """Returns a Document object from a json string representation"""

        return cls.model_validate_json(document)

class Reader(BaseModel):
    chunk: bool = True
    chunk_size: int = 3000
    separators: List[str] = ["\n", "\n\n", "\r", "\r\n", "\n\r", "\t", " ", "  "]

    def read(self, obj: Any) -> List[Document]:
        raise NotImplementedError

    def clean_text(self, text: str) -> str:
        """Clean the text by replacing multiple newlines with a single newline"""
        import re

        # Replace multiple newlines with a single newline
        cleaned_text = re.sub(r"\n+", "\n", text)
        # Replace multiple spaces with a single space
        cleaned_text = re.sub(r"\s+", " ", cleaned_text)
        # Replace multiple tabs with a single tab
        cleaned_text = re.sub(r"\t+", "\t", cleaned_text)
        # Replace multiple carriage returns with a single carriage return
        cleaned_text = re.sub(r"\r+", "\r", cleaned_text)
        # Replace multiple form feeds with a single form feed
        cleaned_text = re.sub(r"\f+", "\f", cleaned_text)
        # Replace multiple vertical tabs with a single vertical tab
        cleaned_text = re.sub(r"\v+", "\v", cleaned_text)

        return cleaned_text

    def chunk_document(self, document: Document) -> List[Document]:
        """Chunk the document content into smaller documents"""
        content = document.content
        cleaned_content = self.clean_text(content)
        content_length = len(cleaned_content)
        chunked_documents: List[Document] = []
        chunk_number = 1
        chunk_meta_data = document.meta_data

        start = 0
        while start < content_length:
            end = start + self.chunk_size

            # Ensure we're not splitting a word in half
            if end < content_length:
                while end > start and cleaned_content[end] not in [" ", "\n", "\r", "\t"]:
                    end -= 1

            # If the entire chunk is a word, then just split it at self.chunk_size
            if end == start:
                end = start + self.chunk_size

            # If the end is greater than the content length, then set it to the content length
            if end > content_length:
                end = content_length

            chunk = cleaned_content[start:end]
            meta_data = chunk_meta_data.copy()
            meta_data["chunk"] = chunk_number
            chunk_id = None
            if document.id:
                chunk_id = f"{document.id}_{chunk_number}"
            elif document.name:
                chunk_id = f"{document.name}_{chunk_number}"
            meta_data["chunk_size"] = len(chunk)
            chunked_documents.append(
                Document(
                    id=chunk_id,
                    name=document.name,
                    meta_data=meta_data,
                    content=chunk,
                )
            )
            chunk_number += 1
            start = end
        return chunked_documents

class WebsiteReader(Reader):
    """Reader for Websites"""

    max_depth: int = 3
    max_links: int = 10

    _visited: Set[str] = set()
    _urls_to_crawl: List[Tuple[str, int]] = []

    def delay(self, min_seconds=1, max_seconds=3):
        """
        Introduce a random delay.

        :param min_seconds: Minimum number of seconds to delay. Default is 1.
        :param max_seconds: Maximum number of seconds to delay. Default is 3.
        """
        sleep_time = random.uniform(min_seconds, max_seconds)
        time.sleep(sleep_time)

    def _get_primary_domain(self, url: str) -> str:
        """
        Extract primary domain from the given URL.

        :param url: The URL to extract the primary domain from.
        :return: The primary domain.
        """
        domain_parts = urlparse(url).netloc.split(".")
        # Return primary domain (excluding subdomains)
        return ".".join(domain_parts[-2:])

    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """
        Extracts the main content from a BeautifulSoup object.

        :param soup: The BeautifulSoup object to extract the main content from.
        :return: The main content.
        """
        # Try to find main content by specific tags or class names
        for tag in ["article", "main"]:
            element = soup.find(tag)
            if element:
                return element.get_text(strip=True, separator=" ")

        for class_name in ["content", "main-content", "post-content"]:
            element = soup.find(class_=class_name)
            if element:
                return element.get_text(strip=True, separator=" ")

        return ""

    def crawl(self, url: str, starting_depth: int = 1) -> Dict[str, str]:
        """
        Crawls a website and returns a dictionary of URLs and their corresponding content.

        Parameters:
        - url (str): The starting URL to begin the crawl.
        - starting_depth (int, optional): The starting depth level for the crawl. Defaults to 1.

        Returns:
        - Dict[str, str]: A dictionary where each key is a URL and the corresponding value is the main
                          content extracted from that URL.

        Note:
        The function focuses on extracting the main content by prioritizing content inside common HTML tags
        like `<article>`, `<main>`, and `<div>` with class names such as "content", "main-content", etc.
        The crawler will also respect the `max_depth` attribute of the WebCrawler class, ensuring it does not
        crawl deeper than the specified depth.
        """
        num_links = 0
        crawler_result: Dict[str, str] = {}
        primary_domain = self._get_primary_domain(url)
        # Add starting URL with its depth to the global list
        self._urls_to_crawl.append((url, starting_depth))
        while self._urls_to_crawl:
            # Unpack URL and depth from the global list
            current_url, current_depth = self._urls_to_crawl.pop(0)

            # Skip if
            # - URL is already visited
            # - does not end with the primary domain,
            # - exceeds max depth
            # - exceeds max links
            if (
                current_url in self._visited
                or not urlparse(current_url).netloc.endswith(primary_domain)
                or current_depth > self.max_depth
                or num_links >= self.max_links
            ):
                continue

            self._visited.add(current_url)
            self.delay()

            try:
                logger.debug(f"Crawling: {current_url}")
                response = httpx.get(current_url, timeout=10)
                soup = BeautifulSoup(response.content, "html.parser")

                # Extract main content
                main_content = self._extract_main_content(soup)
                if main_content:
                    crawler_result[current_url] = main_content
                    num_links += 1

                # Add found URLs to the global list, with incremented depth
                for link in soup.find_all("a", href=True):
                    full_url = urljoin(current_url, link["href"])
                    parsed_url = urlparse(full_url)
                    if parsed_url.netloc.endswith(primary_domain) and not any(
                        parsed_url.path.endswith(ext) for ext in [".pdf", ".jpg", ".png"]
                    ):
                        if full_url not in self._visited and (full_url, current_depth + 1) not in self._urls_to_crawl:
                            self._urls_to_crawl.append((full_url, current_depth + 1))

            except Exception as e:
                logger.debug(f"Failed to crawl: {current_url}: {e}")
                pass

        return crawler_result

    def read(self, url: str) -> List[Document]:
        """
        Reads a website and returns a list of documents.

        This function first converts the website into a dictionary of URLs and their corresponding content.
        Then iterates through the dictionary and returns chunks of content.

        :param url: The URL of the website to read.
        :return: A list of documents.
        """

        logger.debug(f"Reading: {url}")
        crawler_result = self.crawl(url)
        documents = []
        for crawled_url, crawled_content in crawler_result.items():
            if self.chunk:
                documents.extend(
                    self.chunk_document(
                        Document(
                            name=url, id=str(crawled_url), meta_data={"url": str(crawled_url)}, content=crawled_content
                        )
                    )
                )
            else:
                documents.append(
                    Document(
                        name=url,
                        id=str(crawled_url),
                        meta_data={"url": str(crawled_url)},
                        content=crawled_content,
                    )
                )
        return documents

class PDFReader(Reader):
    """Reader for PDF files"""

    def read(self, pdf: Union[str, Path, IO[Any]]) -> List[Document]:
        if not pdf:
            raise ValueError("No pdf provided")

        try:
            from pypdf import PdfReader as DocumentReader  # noqa: F401
        except ImportError:
            raise ImportError("`pypdf` not installed")

        doc_name = ""
        try:
            if isinstance(pdf, str):
                doc_name = pdf.split("/")[-1].split(".")[0].replace(" ", "_")
            else:
                doc_name = pdf.name.split(".")[0]
        except Exception:
            doc_name = "pdf"

        logger.info(f"Reading: {doc_name}")
        doc_reader = DocumentReader(pdf)

        documents = [
            Document(
                name=doc_name,
                id=f"{doc_name}_{page_number}",
                meta_data={"page": page_number},
                content=page.extract_text(),
            )
            for page_number, page in enumerate(doc_reader.pages, start=1)
        ]
        if self.chunk:
            chunked_documents = []
            for document in documents:
                chunked_documents.extend(self.chunk_document(document))
            return chunked_documents
        return documents

class PDFUrlReader(Reader):
    """Reader for PDF files from URL"""

    def read(self, url: str) -> List[Document]:
        if not url:
            raise ValueError("No url provided")

        from io import BytesIO

        try:
            import httpx
        except ImportError:
            raise ImportError("`httpx` not installed")

        try:
            from pypdf import PdfReader as DocumentReader  # noqa: F401
        except ImportError:
            raise ImportError("`pypdf` not installed")

        logger.info(f"Reading: {url}")
        response = httpx.get(url)

        doc_name = url.split("/")[-1].split(".")[0].replace("/", "_").replace(" ", "_")
        doc_reader = DocumentReader(BytesIO(response.content))

        documents = [
            Document(
                name=doc_name,
                id=f"{doc_name}_{page_number}",
                meta_data={"page": page_number},
                content=page.extract_text(),
            )
            for page_number, page in enumerate(doc_reader.pages, start=1)
        ]
        if self.chunk:
            chunked_documents = []
            for document in documents:
                chunked_documents.extend(self.chunk_document(document))
            return chunked_documents
        return documents

class PDFImageReader(Reader):
    """Reader for PDF files with text and images extraction"""

    def read(self, pdf: Union[str, Path, IO[Any]]) -> List[Document]:
        if not pdf:
            raise ValueError("No pdf provided")

        try:
            import rapidocr_onnxruntime as rapidocr
            from pypdf import PdfReader as DocumentReader  # noqa: F401
        except ImportError:
            raise ImportError("`pypdf` or `rapidocr_onnxruntime` not installed")

        doc_name = ""
        try:
            if isinstance(pdf, str):
                doc_name = pdf.split("/")[-1].split(".")[0].replace(" ", "_")
            else:
                doc_name = pdf.name.split(".")[0]
        except Exception:
            doc_name = "pdf"

        logger.info(f"Reading: {doc_name}")
        doc_reader = DocumentReader(pdf)

        # Initialize RapidOCR
        ocr = rapidocr.RapidOCR()

        documents = []
        for page_number, page in enumerate(doc_reader.pages, start=1):
            page_text = page.extract_text() or ""
            images_text_list: List = []

            for image_object in page.images:
                image_data = image_object.data

                # Perform OCR on the image
                ocr_result, elapse = ocr(image_data)

                # Extract text from OCR result
                if ocr_result:
                    images_text_list += [item[1] for item in ocr_result]

            images_text: str = "\n".join(images_text_list)
            content = page_text + "\n" + images_text

            documents.append(
                Document(
                    name=doc_name,
                    id=f"{doc_name}_{page_number}",
                    meta_data={"page": page_number},
                    content=content,
                )
            )

        if self.chunk:
            chunked_documents = []
            for document in documents:
                chunked_documents.extend(self.chunk_document(document))
            return chunked_documents

        return documents

class PDFUrlImageReader(Reader):
    """Reader for PDF files from URL with text and images extraction"""

    def read(self, url: str) -> List[Document]:
        if not url:
            raise ValueError("No url provided")

        from io import BytesIO

        try:
            import httpx
            from pypdf import PdfReader as DocumentReader
            import rapidocr_onnxruntime as rapidocr
        except ImportError:
            raise ImportError("`httpx`, `pypdf` or `rapidocr_onnxruntime` not installed")

        # Read the PDF from the URL
        logger.info(f"Reading: {url}")
        response = httpx.get(url)

        doc_name = url.split("/")[-1].split(".")[0].replace(" ", "_")
        doc_reader = DocumentReader(BytesIO(response.content))

        # Initialize RapidOCR
        ocr = rapidocr.RapidOCR()

        # Process each page of the PDF
        documents = []
        for page_number, page in enumerate(doc_reader.pages, start=1):
            page_text = page.extract_text() or ""
            images_text_list = []

            # Extract and process images
            for image_object in page.images:
                image_data = image_object.data

                # Perform OCR on the image
                ocr_result, elapse = ocr(image_data)

                # Extract text from OCR result
                if ocr_result:
                    images_text_list += [item[1] for item in ocr_result]

            images_text = "\n".join(images_text_list)
            content = page_text + "\n" + images_text

            # Append the document
            documents.append(
                Document(
                    name=doc_name,
                    id=f"{doc_name}_{page_number}",
                    meta_data={"page": page_number},
                    content=content,
                )
            )

        # Optionally chunk documents
        if self.chunk:
            chunked_documents = []
            for document in documents:
                chunked_documents.extend(self.chunk_document(document))
            return chunked_documents

        return documents
