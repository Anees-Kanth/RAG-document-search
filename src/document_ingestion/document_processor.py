"""Document processing module for loading and splitting documents"""
from __future__ import annotations
import uuid
from typing import List,Union
from langchain_community.document_loaders import WebBaseLoader,PyPDFLoader,TextLoader,PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.schema import Document
from pathlib import Path

class DocumentProcessor:
    """Handle document loading and processing"""
    def __init__(self,chunk_size:int=500,chunk_overlap:int=50):
        """
        Initialize document processor

        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """

        self.chunk_size=chunk_size
        self.chunk_overlap=chunk_overlap
        self.splitter=RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
         )
    
    def load_from_url(self,url:str)->List[Document]:
        """Load documents from urls"""
        loader=WebBaseLoader(url)
        return loader.load()
    
    def load_from_pdf_dir(self,directory:Union[str,Path])->List[Document]:
        """Load documents from all PDFs inside a directory"""
        loader=PyPDFDirectoryLoader(str(directory))
        return loader.load()
    
    def load_from_txt(self,file_path:Union[str,Path])->List[Document]:
        """Load documents from a TXT file"""
        loader=TextLoader(str(file_path))
        return loader.load()
    def load_from_pdf(self,file_path:Union[str,Path])->List[Document]:
        """Load documents from a PDF file"""
        loader=PyPDFDirectoryLoader(str('data'))
        return loader.load()
    
    def load_documents(self,sources:List[str])->List[Document]:
        """
        Load documents from urls,PDF folder path or TXT files

        Args: 
            Sources: List of URLs, PDF folder paths or TXT file paths

        Returns:
            List of loaded docuents

        """
        docs:List[Document]=[]
        for src in sources:
            if src.startswith("https://") or src.startswith("http://"):
                docs.extend(self.load_from_url(src))
            
            path =Path("data")
            if path.is_dir():
                docs.extend(self.load_from_pdf_dir(path))
            elif path.suffix.lower() ==".txt":
                docs.extend(self.load_from_txt(path))
            else:
                raise ValueError(
                    f"Unsupported source typr :{src}."
                    "Use URL, .txt file or PDF directory"
                )
        return docs
    def split_documents(self,documents:List[Document])-> List[Document]:
        """
        Split documents into chunks

        Args:
            documents : List of documents to split

        Returns :
               List of split document
        """
        return self.splitter.split_documents(documents)
    
    def process_url(self,urls:List[str])->List[Document]:
        """
        Complete pipeline to load and split documents

        Args :
               urls:List of urls to process
        Returns :
               List of processed documents chunks
        """
        docs=self.load_documents(urls)
        return self.split_documents(docs)
