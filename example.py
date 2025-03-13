"""
Example script demonstrating how to use the RAG pipeline programmatically.
"""

import os
import sys
from dotenv import load_dotenv

from rag.rag_pipeline import RAGPipeline

# Load environment variables
load_dotenv()


def main():
    """
    Main function to demonstrate the RAG pipeline.
    """
    # Check if OpenAI API key is set
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is not set.")
        print("Please set it in a .env file or export it in your shell.")
        sys.exit(1)

    # Initialize the RAG pipeline
    print("Initializing RAG pipeline...")
    rag_pipeline = RAGPipeline(
        embedding_model_type="openai",
        llm_model_name="gpt-3.5-turbo",
        persist_directory="data/chroma_db",
    )

    # Process a document or text
    while True:
        print("\nOptions:")
        print("1. Process a document")
        print("2. Process text input")
        print("3. Ask a question")
        print("4. Clear all documents")
        print("5. Exit")

        choice = input("\nEnter your choice (1-5): ")

        if choice == "1":
            # Process a document
            file_path = input("Enter the path to the document: ")
            if os.path.exists(file_path):
                print(f"Processing document: {file_path}")
                rag_pipeline.add_document(file_path)
                print("Document processed successfully!")
            else:
                print(f"Error: File not found at {file_path}")

        elif choice == "2":
            # Process text input
            text = input("Enter the text content: ")
            source_name = input("Enter a name for this text (for reference): ")

            if text:
                print("Processing text...")
                metadata = {"source": source_name or "User Input"}
                rag_pipeline.add_text(text, metadata)
                print("Text processed successfully!")
            else:
                print("Error: Text content cannot be empty.")

        elif choice == "3":
            # Ask a question
            question = input("Enter your question: ")

            if question:
                print("\nThinking...")
                response = rag_pipeline.query(question)

                print("\nAnswer:")
                print(response["answer"])

                print("\nSources:")
                for i, doc in enumerate(response["source_documents"]):
                    source = doc.metadata.get("source", f"Document {i+1}")
                    print(f"\nSource {i+1}: {source}")
                    print(f"Content: {doc.page_content[:200]}...")
            else:
                print("Error: Question cannot be empty.")

        elif choice == "4":
            # Clear all documents
            confirm = input("Are you sure you want to clear all documents? (y/n): ")
            if confirm.lower() == "y":
                print("Clearing all documents...")
                rag_pipeline.clear()
                print("All documents cleared!")

        elif choice == "5":
            # Exit
            print("Exiting...")
            break

        else:
            print("Invalid choice. Please enter a number between 1 and 5.")


if __name__ == "__main__":
    main()
