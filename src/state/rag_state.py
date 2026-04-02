
from __future__ import annotations
import uuid
"""RAG state defination for LanGraph"""
from typing import List
from pydantic import BaseModel
from langchain_classic.schema import Document

class RAGState(BaseModel):
    question:str
    retrieved_docs:List[Document]=[]
    answer:str=''