"""
Retriever module for finding relevant document chunks.
"""

import os
from typing import List, Dict, Any, Optional
import numpy as np
from langchain.docstore.document import Document
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions


class VectorStoreRetriever:
    """
    Class for retrieving relevant document chunks using vector similarity.
    """

    def __init__(
        self,
        collection_name: str = "document_collection",
        embedding_function_type: str = "openai",
        persist_directory: Optional[str] = None,
    ):
        """
        Initialize the vector store retriever.

        Args:
            collection_name: Name of the vector store collection
            embedding_function_type: Type of embedding function to use ('openai' or 'huggingface')
            persist_directory: Directory to persist the vector store (if None, in-memory store is used)
        """
        self.collection_name = collection_name

        # Set up ChromaDB client
        if persist_directory:
            self.client = chromadb.PersistentClient(
                path=persist_directory, settings=Settings(anonymized_telemetry=False)
            )
        else:
            self.client = chromadb.Client(Settings(anonymized_telemetry=False))

        # Set up embedding function
        if embedding_function_type == "openai":
            self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.environ.get("OPENAI_API_KEY"),
                model_name="text-embedding-ada-002",
            )
        elif embedding_function_type == "huggingface":
            self.embedding_function = (
                embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name="all-mpnet-base-v2"
                )
            )
        else:
            raise ValueError(
                f"Unsupported embedding function type: {embedding_function_type}"
            )

        # Create or get collection
        try:
            self.collection = self.client.get_collection(
                name=collection_name, embedding_function=self.embedding_function
            )
        except:
            self.collection = self.client.create_collection(
                name=collection_name, embedding_function=self.embedding_function
            )

    def add_documents(
        self, documents: List[Document], embeddings: Optional[List[List[float]]] = None
    ):
        """
        Add documents to the vector store.

        Args:
            documents: List of Document objects
            embeddings: Optional pre-computed embeddings
        """
        ids = [f"doc_{i}" for i in range(len(documents))]
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        self.collection.add(
            ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings
        )

    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: Query string
            top_k: Number of documents to retrieve

        Returns:
            List of relevant Document objects
        """
        results = self.collection.query(query_texts=[query], n_results=top_k)

        documents = []
        for i in range(len(results["documents"][0])):
            doc = Document(
                page_content=results["documents"][0][i],
                metadata=results["metadatas"][0][i],
            )
            documents.append(doc)

        return documents

    def clear(self):
        """
        Clear all documents from the collection.
        """
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name, embedding_function=self.embedding_function
        )
