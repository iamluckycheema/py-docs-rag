"""
RAG pipeline module for combining all components into a complete system.
"""

import os
from typing import List, Dict, Any, Optional
from langchain.docstore.document import Document

from rag.document_processor import DocumentProcessor
from rag.embeddings import EmbeddingManager
from rag.retriever import VectorStoreRetriever
from rag.generator import ResponseGenerator


class RAGPipeline:
    """
    Complete RAG pipeline that combines document processing, embedding, retrieval, and generation.
    """

    def __init__(
        self,
        embedding_model_type: str = "openai",
        llm_model_name: str = "gpt-4o-mini",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        collection_name: str = "document_collection",
        persist_directory: Optional[str] = None,
    ):
        """
        Initialize the RAG pipeline.

        Args:
            embedding_model_type: Type of embedding model to use ('openai' or 'huggingface')
            llm_model_name: Name of the LLM model to use
            chunk_size: Size of document chunks
            chunk_overlap: Overlap between chunks
            collection_name: Name of the vector store collection
            persist_directory: Directory to persist the vector store
        """
        self.document_processor = DocumentProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        self.embedding_manager = EmbeddingManager(
            model_type=embedding_model_type,
        )

        self.retriever = VectorStoreRetriever(
            collection_name=collection_name,
            embedding_function_type=embedding_model_type,
            persist_directory=persist_directory,
        )

        self.generator = ResponseGenerator(
            model_name=llm_model_name,
        )

    def add_document(self, file_path: str):
        """
        Process a document and add it to the retrieval system.

        Args:
            file_path: Path to the document file
        """
        # Load and process the document
        chunked_documents = self.document_processor.load_and_process(file_path)

        # Embed the documents
        embedded_docs = self.embedding_manager.embed_documents(chunked_documents)

        # Add to the retriever
        self.retriever.add_documents(
            documents=embedded_docs["documents"],
            embeddings=embedded_docs["embeddings"],
        )

    def add_text(self, text: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Process text and add it to the retrieval system.

        Args:
            text: Raw text content
            metadata: Optional metadata for the document
        """
        # Process the text
        chunked_documents = self.document_processor.process_text(text, metadata)

        # Embed the documents
        embedded_docs = self.embedding_manager.embed_documents(chunked_documents)

        # Add to the retriever
        self.retriever.add_documents(
            documents=embedded_docs["documents"],
            embeddings=embedded_docs["embeddings"],
        )

    def query(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Query the RAG system with a question.

        Args:
            question: Question to answer
            top_k: Number of documents to retrieve

        Returns:
            Dictionary with the response and metadata
        """
        # Retrieve relevant documents
        retrieved_docs = self.retriever.retrieve(question, top_k=top_k)

        # Generate a response
        response = self.generator.generate_response(
            question=question,
            documents=retrieved_docs,
        )

        return response

    def clear(self):
        """
        Clear all documents from the retrieval system.
        """
        self.retriever.clear()
