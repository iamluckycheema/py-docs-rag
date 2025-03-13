# Document RAG System

This is a Retrieval-Augmented Generation (RAG) system that processes documents and allows querying information from them using natural language.

## Features

- Document ingestion and processing (PDF, TXT, DOCX)
- Vector embedding of document chunks
- Semantic search for relevant information
- LLM-powered question answering with context from documents
- Simple web interface for interacting with the system

## Setup

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your OpenAI API key (copy from `.env.example`):
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage

1. Run the Streamlit app:
   ```
   streamlit run app.py
   ```
2. Upload your document(s) through the web interface or enter text directly
3. Ask questions about the content of your documents

## Project Structure

- `app.py`: Main Streamlit application
- `rag/`: Core RAG system components
  - `document_processor.py`: Document loading and chunking
  - `embeddings.py`: Vector embedding functionality
  - `retriever.py`: Retrieval system for finding relevant chunks
  - `generator.py`: LLM-based answer generation
  - `rag_pipeline.py`: Complete RAG pipeline

## How It Works

1. **Document Processing**: Documents are loaded and split into manageable chunks with some overlap to maintain context.
2. **Embedding**: Each chunk is converted into a vector representation using embeddings (OpenAI or HuggingFace).
3. **Storage**: These vectors are stored in a vector database (ChromaDB).
4. **Retrieval**: When a question is asked, the system finds the most relevant chunks by comparing the question's embedding with the stored document embeddings.
5. **Generation**: The relevant chunks are sent to an LLM along with the question to generate a contextually accurate answer.

## Customization

You can customize the RAG system by modifying the parameters in the `RAGPipeline` initialization:

- `embedding_model_type`: Choose between "openai" or "huggingface" for embeddings
- `llm_model_name`: Specify which OpenAI model to use (e.g., "gpt-3.5-turbo", "gpt-4")
- `chunk_size`: Control how large each document chunk should be
- `chunk_overlap`: Set the overlap between chunks to maintain context
- `persist_directory`: Specify where to store the vector database

## Troubleshooting

- **API Key Issues**: Ensure your OpenAI API key is correctly set in the `.env` file
- **Memory Issues**: For large documents, you may need to reduce the chunk size
- **Performance**: For faster processing, consider using a local embedding model with HuggingFace

## License

This project is open source and available under the MIT License.
