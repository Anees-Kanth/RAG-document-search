"""Langgraph nodes for workflow"""
from __future__ import annotations
import uuid
from src.state.rag_state import RAGState
import uuid


class RAGNodes:
    """Contains node functions for RAG workflow"""
    def __init__(self,retriever,llm):
        """
        Iniialize RAG node
        Args :
            retriever : Document retriever instance
            llm : Language model instance
        """
        self.retriever=retriever
        self.llm=llm

    def retrieve_docs(self,state:RAGState)->RAGState:
        """
        Retrieve relevant documents node

        Args: current RAG State

        Returns :
        Updated RAG state with retrieved documents
        """
        docs=self.retriever.invoke(state.question)
        return RAGState(question=state.question,retrieved_docs=docs)
    
    def generate_answer(self,state:RAGState)->RAGState:
        """
        Generate answer from retrieved documents

        Args:
            state: Current RAG state with retrieved documents
        Returns:
             Updated RAG state with generated answer
        """
        context="\n\n".join([doc.page_content for doc in state.retrieved_docs])

        prompt=f"""Answer the question based on the context.
Context:
{context}

Question: {state.question}
"""   
        response=self.llm.invoke(prompt)

        return RAGState(
            question=state.question,
            retrieved_docs=state.retrieved_docs,
            answer=response.context
        )