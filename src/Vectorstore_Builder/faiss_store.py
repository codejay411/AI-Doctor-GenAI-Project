from langchain_community.vectorstores import FAISS
import os


class FAISSStore:
    def __init__(self, embedding_model, index_path="faiss_index"):
        self.embedding_model = embedding_model
        self.vectorstore = None
        self.index_path = index_path

    def build(self, documents):
        """Build FAISS index from documents."""
        self.vectorstore = FAISS.from_documents(
            documents=documents,
            embedding=self.embedding_model
        )
        return self.vectorstore

    def save(self):
        """Save FAISS index and metadata to local folder."""
        if self.vectorstore is None:
            raise ValueError("Vectorstore not built. Cannot save.")

        os.makedirs(self.index_path, exist_ok=True)
        self.vectorstore.save_local(self.index_path)
        print(f"✔ FAISS index saved at: {self.index_path}")

    def load(self):
        """Load FAISS index from local folder."""
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"No FAISS index found in {self.index_path}")

        self.vectorstore = FAISS.load_local(
            folder_path=self.index_path,
            embeddings=self.embedding_model,
            allow_dangerous_deserialization=True
        )
        print(f"✔ FAISS index loaded from: {self.index_path}")
        return self.vectorstore

    def get_retriever(self, vectordb):
        """Return retriever object."""
        if vectordb is None:
            raise ValueError("Vectorstore not initialized")
        return vectordb.as_retriever()
