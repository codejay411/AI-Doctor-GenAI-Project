from langchain_community.document_loaders import PyPDFLoader
import os


class PDFLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self):
        """Load and return PDF as a list of LangChain Document objects."""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"PDF not found: {self.file_path}")
        
        loader = PyPDFLoader(self.file_path)
        return loader.load()
