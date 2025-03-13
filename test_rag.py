"""
Test script for the RAG system.
"""

import os
import unittest
from dotenv import load_dotenv

from rag.document_processor import DocumentProcessor
from rag.embeddings import EmbeddingManager
from rag.retriever import VectorStoreRetriever
from rag.generator import ResponseGenerator
from rag.rag_pipeline import RAGPipeline

# Load environment variables
load_dotenv()


class TestRAGSystem(unittest.TestCase):
    """
    Test cases for the RAG system.
    """

    def setUp(self):
        """
        Set up test environment.
        """
        # Skip tests if OpenAI API key is not set
        if not os.environ.get("OPENAI_API_KEY"):
            self.skipTest("OPENAI_API_KEY environment variable is not set")

        # Create test data directory
        os.makedirs("test_data", exist_ok=True)

        # Create a test text file
        self.test_file_path = "test_data/test_document.txt"
        with open(self.test_file_path, "w") as f:
            f.write(
                """
            This is a test document about artificial intelligence.
            
            Artificial intelligence (AI) is intelligence demonstrated by machines, 
            as opposed to intelligence displayed by animals including humans.
            
            Machine learning is a subset of AI that focuses on the development of algorithms 
            that can access data and use it to learn for themselves.
            
            Deep learning is a subset of machine learning that uses neural networks 
            with many layers to analyze various factors of data.
            
            Natural Language Processing (NLP) is a field of AI that gives machines 
            the ability to read, understand, and derive meaning from human languages.
            """
            )

    def tearDown(self):
        """
        Clean up after tests.
        """
        # Remove test file
        if os.path.exists(self.test_file_path):
            os.remove(self.test_file_path)

        # Remove test data directory if empty
        try:
            os.rmdir("test_data")
        except:
            pass

        # Remove test vector store
        if os.path.exists("test_data/chroma_db"):
            import shutil

            shutil.rmtree("test_data/chroma_db")

    def test_document_processor(self):
        """
        Test document processing functionality.
        """
        processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)
        documents = processor.load_and_process(self.test_file_path)

        self.assertGreater(len(documents), 0, "Document should be split into chunks")
        self.assertEqual(documents[0].metadata["source"], self.test_file_path)

    def test_embedding_manager(self):
        """
        Test embedding functionality.
        """
        embedding_manager = EmbeddingManager(model_type="openai")

        # Test single query embedding
        query_embedding = embedding_manager.get_query_embedding(
            "What is artificial intelligence?"
        )
        self.assertIsInstance(query_embedding, list)
        self.assertGreater(len(query_embedding), 0)

        # Test document embeddings
        texts = ["This is a test document.", "Another test document."]
        embeddings = embedding_manager.get_embeddings(texts)
        self.assertEqual(len(embeddings), 2)

    def test_rag_pipeline(self):
        """
        Test the complete RAG pipeline.
        """
        # Initialize RAG pipeline
        rag_pipeline = RAGPipeline(
            embedding_model_type="openai",
            llm_model_name="gpt-3.5-turbo",
            persist_directory="test_data/chroma_db",
        )

        # Add document
        rag_pipeline.add_document(self.test_file_path)

        # Test query
        response = rag_pipeline.query("What is machine learning?")

        self.assertIn("question", response)
        self.assertIn("answer", response)
        self.assertIn("source_documents", response)
        self.assertGreater(len(response["source_documents"]), 0)

        # Clear documents
        rag_pipeline.clear()


if __name__ == "__main__":
    unittest.main()
