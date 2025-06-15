from langchain.embeddings import HuggingFaceEmbeddings
from loader import process_documents
from langchain.vectorstores import Chroma
from chromadb import Client
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cpu"}  # ou "cuda" si tu as un GPU nvidia
)

chroma_client = Client(Settings(persist_directory="./chroma_db", chroma_db_impl="duckdb+parquet"))
collection = chroma_client.get_collection(name="chatbot-rag")

embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

query = "Quelle est la fourchette du JEH ?"

query_embedding = embedder.encode(query).tolist()

results = collection.query(query_embeddings=[query_embedding], n_results=3)

