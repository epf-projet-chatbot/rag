from chatbot_rag.loader import process_documents
from langchain_chroma import Chroma
from langchain_core.documents import Document
import getpass
import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load environment variables first
load_dotenv()

CHROMA_PATH = os.getenv("CHROMA_PATH", "chroma_db")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialisation des embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

def embed(text: str) -> list[float]:
    """
    Vectorise un texte en utilisant Google Generative AI Embeddings.
    
    Args:
        text (str): Le texte à vectoriser.
        
    Returns:
        list[float]: Le vecteur d'embedding du texte.
    """
    try:
        # Utilisation de l'API LangChain pour générer l'embedding
        result = embeddings.embed_query(text)
        return result
    except Exception as e:
        print(f"Erreur lors de la vectorisation : {e}")
        return []

def add_to_chroma(chunks: list[Document]):
    """
    Ajoute des chunks à la base de données Chroma.
    
    Args:
        chunks (list[Document]): Liste de documents à ajouter.
    """
    try:
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
        max =db._client.get_max_batch_size()
        # Obtenir les IDs existants pour éviter les doublons
        existing_ids = set()
        try:
            existing_collection = db.get()
            if existing_collection and 'ids' in existing_collection:
                existing_ids = set(existing_collection['ids'])
        except:
            pass
        
        chunk_ids = []
        
        for i, chunk in enumerate(chunks):
            source = chunk.metadata.get('source', 'unknown')
            page = chunk.metadata.get('page', 0)
            
            # Format: document_id:page_id:chunk_index
            chunk_id = f"{source}:{page}:{i}"
            
            # S'assurer qu'il n'y a pas de doublons
            counter = 0
            original_chunk_id = chunk_id
            while chunk_id in existing_ids or chunk_id in chunk_ids:
                counter += 1
                chunk_id = f"{original_chunk_id}_{counter}"
            
            chunk.metadata['id'] = chunk_id
            chunk_ids.append(chunk_id)
            existing_ids.add(chunk_id)
        
        # Ajouter les documents par paquets de taille max
        for i in range(0, len(chunks), max):
            batch_chunks = chunks[i:i + max]
            batch_ids = chunk_ids[i:i + max]
            db.add_documents(batch_chunks, ids=batch_ids)
            print(f"{len(chunks)} chunks ajoutés à la base de données Chroma.")
        
    except Exception as e:
        print(f"Erreur lors de l'ajout à Chroma : {e}")

if __name__ == "__main__":
    """
    Point d'entrée pour le script.
    Charge les documents, les prétraite, les vectorise et les ajoute à la base de données Chroma.
    """
    
    # Chargement et ajout des documents
    data_path = "./data/data_complete"
    if not os.path.exists(data_path):
        print(f"Erreur : Le répertoire {data_path} n'existe pas.")
        exit(1)
    
    documents = process_documents(data_path)
    if documents:
        add_to_chroma(documents)
        print("Chroma DB mise à jour avec succès.")
    else:
        print("Aucun document trouvé à traiter.")