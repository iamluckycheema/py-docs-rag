"""
Embeddings module for creating vector representations of text.
"""

import os
from typing import List, Dict, Any
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document


class EmbeddingManager:
    """
    Class for managing text embeddings.
    """

    def __init__(self, model_type: str = "openai"):
        """
        Initialize the embedding manager.

        Args:
            model_type: Type of embedding model to use ('openai' or 'huggingface')
        """
        self.model_type = model_type

        if model_type == "openai":
            self.embedding_model = OpenAIEmbeddings(
                model="text-embedding-ada-002",
            )
        elif model_type == "huggingface":
            self.embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2"
            )
        else:
            raise ValueError(f"Unsupported embedding model type: {model_type}")

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for a list of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        return self.embedding_model.embed_documents(texts)

    def get_query_embedding(self, query: str) -> List[float]:
        """
        Get embedding for a single query text.

        Args:
            query: Query text to embed

        Returns:
            Embedding vector
        """
        return self.embedding_model.embed_query(query)

    def embed_documents(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Embed a list of documents and return documents with embeddings.

        Args:
            documents: List of Document objects

        Returns:
            Dictionary with documents and their embeddings
        """
        texts = [doc.page_content for doc in documents]
        embeddings = self.get_embeddings(texts)

        return {"documents": documents, "embeddings": embeddings}
