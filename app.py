"""
Streamlit app for the RAG system.
"""

import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from rag.rag_pipeline import RAGPipeline

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="Document RAG System",
    page_icon="📚",
    layout="wide",
)

# Initialize session state
if "rag_pipeline" not in st.session_state:
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)

    st.session_state.rag_pipeline = RAGPipeline(
        embedding_model_type="openai",
        llm_model_name="gpt-4o-mini",
        persist_directory="data/chroma_db",
    )

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# App title
st.title("📚 Document RAG System")

# Sidebar for document upload and settings
with st.sidebar:
    st.header("Document Upload")

    uploaded_file = st.file_uploader(
        "Upload a document (PDF, TXT, DOCX)",
        type=["pdf", "txt", "docx"],
    )

    if uploaded_file is not None:
        with st.spinner("Processing document..."):
            # Save the uploaded file to a temporary file
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}"
            ) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            # Process the document
            st.session_state.rag_pipeline.add_document(tmp_file_path)

            # Clean up the temporary file
            os.unlink(tmp_file_path)

            st.success(f"Document '{uploaded_file.name}' processed successfully!")

    st.header("Text Input")

    text_input = st.text_area("Or enter text directly:")
    text_name = st.text_input("Text name (for reference):")

    if st.button("Process Text") and text_input:
        with st.spinner("Processing text..."):
            metadata = {"source": text_name or "User Input"}
            st.session_state.rag_pipeline.add_text(text_input, metadata)
            st.success("Text processed successfully!")

    if st.button("Clear All Documents"):
        with st.spinner("Clearing documents..."):
            st.session_state.rag_pipeline.clear()
            st.session_state.chat_history = []
            st.success("All documents cleared!")

# Main area for chat
st.header("Chat with your Documents")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

        # Display sources if available
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("Sources"):
                for i, source in enumerate(message["sources"]):
                    st.markdown(f"**Source {i+1}:** {source['source']}")
                    st.text(source["content"])

# Chat input
query = st.chat_input("Ask a question about your documents...")

if query:
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": query})

    # Display user message
    with st.chat_message("user"):
        st.write(query)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.rag_pipeline.query(query)

            st.write(response["answer"])

            # Prepare sources for display
            sources = []
            for doc in response["source_documents"]:
                source = doc.metadata.get("source", "Unknown")
                sources.append(
                    {
                        "source": source,
                        "content": doc.page_content,
                    }
                )

            # Display sources
            with st.expander("Sources"):
                for i, source in enumerate(sources):
                    st.markdown(f"**Source {i+1}:** {source['source']}")
                    st.text(source["content"])

    # Add assistant message to chat history
    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": response["answer"],
            "sources": sources,
        }
    )
