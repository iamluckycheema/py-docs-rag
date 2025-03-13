"""
Generator module for creating responses using an LLM.
"""

import os
from typing import List, Dict, Any, Optional
from langchain.docstore.document import Document
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate


class ResponseGenerator:
    """
    Class for generating responses using an LLM with context from retrieved documents.
    """

    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.0,
    ):
        """
        Initialize the response generator.

        Args:
            model_name: Name of the LLM model to use
            temperature: Temperature parameter for the LLM
        """
        self.model_name = model_name
        self.temperature = temperature

        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
        )

        self.prompt_template = ChatPromptTemplate.from_template(
            """You are a helpful assistant that answers questions based on the provided context.
            
Context:
{context}

Question: {question}

Instructions:
1. Answer the question based only on the provided context.
2. If the context doesn't contain the information needed to answer the question, say "I don't have enough information to answer this question."
3. Keep your answer concise and to the point.
4. If appropriate, cite specific parts of the context that support your answer.

Answer:"""
        )

        # Create a runnable sequence using the new pipe operator
        self.chain = self.prompt_template | self.llm

    def format_context(self, documents: List[Document]) -> str:
        """
        Format a list of documents into a context string.

        Args:
            documents: List of Document objects

        Returns:
            Formatted context string
        """
        context_parts = []

        for i, doc in enumerate(documents):
            source = doc.metadata.get("source", f"Document {i+1}")
            context_parts.append(f"[{source}]\n{doc.page_content}\n")

        return "\n".join(context_parts)

    def generate_response(
        self,
        question: str,
        documents: List[Document],
    ) -> Dict[str, Any]:
        """
        Generate a response to a question using the provided documents as context.

        Args:
            question: Question to answer
            documents: List of Document objects to use as context

        Returns:
            Dictionary with the response and metadata
        """
        context = self.format_context(documents)

        # Use the new invoke method with the runnable sequence
        response = self.chain.invoke(
            {
                "context": context,
                "question": question,
            }
        )

        return {
            "question": question,
            "answer": response.content,
            "source_documents": documents,
        }
