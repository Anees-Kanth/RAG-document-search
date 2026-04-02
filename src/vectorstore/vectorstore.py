"""Vector store module for document embedding and retrieval"""
from __future__ import annotations
import uuid
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_classic.schema import Document

class VectorStore:
    """Manage vector store application"""
    def __init__(self):
        self.embedding=OpenAIEmbeddings()
        self.vectorstore=None
        self.retriever=None

    def create_retriever(self,documents : List[Document]):
        """Create vector store from documents"""
        self.vectorstore=FAISS.from_documents(documents,self.embedding)
        self.retriever=self.vectorstore.as_retriever()

    def get_retriever(self):
        """Get the retriever instance"""

        if self.retriever is None:
            raise ValueError("Vector store not initialized.call create vector first")
        return self.retriever
    def retrieve(self,query:str,k:int=4)->List[Document]:
        """
        Retrieve relevant documents from query
        Args:
           query:Search query
           k:number of documents to retrieve
        Returns:
           List of relevant documents
        """
        if self.retriever is None:
            raise ValueError("Vector store not initialized. call create vector first ")
        return self.retriever.invoke(query)
        

