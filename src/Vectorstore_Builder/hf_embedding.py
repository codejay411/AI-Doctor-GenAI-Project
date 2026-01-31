from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()
import os 

class HFEmbedding:
    def __init__(self, model_name="BAAI/bge-large-en-v1.5", device="cpu"):
        self.model_name = os.getenv("EMBEDDING_MODEL")
        self.device = device

    def load(self):
        """Load Hugging Face embedding model wrapper for LangChain."""
        return HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": self.device},
            encode_kwargs={"normalize_embeddings": True}
        )
