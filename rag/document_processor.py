"""
Document processor module for loading and chunking documents.
"""

import os
from typing import List, Dict, Any, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)
from langchain.docstore.document import Document


class DocumentProcessor:
    """
    Class for loading and processing documents into chunks.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        """
        Initialize the document processor.

        Args:
            chunk_size: Size of each document chunk
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

    def load_document(self, file_path: str) -> List[Document]:
        """
        Load a document from a file path.

        Args:
            file_path: Path to the document

        Returns:
            List of Document objects
        """
        _, file_extension = os.path.splitext(file_path)

        if file_extension.lower() == ".pdf":
            loader = PyPDFLoader(file_path)
        elif file_extension.lower() == ".txt":
            loader = TextLoader(file_path)
        elif file_extension.lower() in [".docx", ".doc"]:
            loader = Docx2txtLoader(file_path)
        else:
            raise ValueError(f"Unsupported file extension: {file_extension}")

        return loader.load()

    def process_documents(self, documents: List[Document]) -> List[Document]:
        """
        Process documents by splitting them into chunks.

        Args:
            documents: List of Document objects

        Returns:
            List of chunked Document objects
        """
        return self.text_splitter.split_documents(documents)

    def load_and_process(self, file_path: str) -> List[Document]:
        """
        Load and process a document in one step.

        Args:
            file_path: Path to the document

        Returns:
            List of chunked Document objects
        """
        documents = self.load_document(file_path)
        return self.process_documents(documents)

    def process_text(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Process raw text into document chunks.

        Args:
            text: Raw text content
            metadata: Optional metadata for the document

        Returns:
            List of chunked Document objects
        """
        if metadata is None:
            metadata = {}

        doc = Document(page_content=text, metadata=metadata)
        return self.text_splitter.split_documents([doc])
