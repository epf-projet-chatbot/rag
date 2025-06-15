from loader import process_documents
from langchain_chroma import Chroma
from langchain_core.documents import Document
import getpass
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings

CHROMA_PATH = "chroma_db"

if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")

# Initialisation des embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

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
        
        db.add_documents(chunks, ids=chunk_ids)
        print(f"{len(chunks)} chunks ajoutés à la base de données Chroma.")
        
    except Exception as e:
        print(f"Erreur lors de l'ajout à Chroma : {e}")

if __name__ == "__main__":
    """
    Point d'entrée pour le script.
    Charge les documents, les prétraite, les vectorise et les ajoute à la base de données Chroma.
    """
    # Test de la fonction embed
    test_text = "Quelle est la fourchette du JEH ?"
    embedding_vector = embed(test_text)
    print(f"Embedding pour '{test_text}': {len(embedding_vector)} dimensions")
    
    # Chargement et ajout des documents
    documents = process_documents("data")
    add_to_chroma(documents)
    
    print("Chroma DB mise à jour avec succès.")