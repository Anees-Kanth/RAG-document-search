"""Confuguration module for Agentic RAG system"""

import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model


load_dotenv()

class Config:
## API Keys
    OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")

## Model configuration
    LLM_MODEL="openai:gpt-4o"

## Document processing
    CHUNK_SIZE=500
    CHUNK_OVERLAP=50

## Default URLS
    DEFAULT_URLS=[
          "https://lilianweng.github.io/posts/2023-06-23-agent/",
           "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/"
     ]

    @classmethod
    def get_llm(cls):
      "Initialize and return LLM model"

      os.environ['OPENAI_API_KEY']=cls.OPENAI_API_KEY
      return init_chat_model(cls.LLM_MODEL)