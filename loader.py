from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, JSONLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
import unicodedata
from typing import List
from langchain.schema import Document
import spacy

# Load spacy model
nlp = spacy.load("fr_core_news_sm")

def load_documents(folder_path):
    """
    Charge tous les documents d'un dossier donné, en parcourant récursivement
    les sous-dossiers et en utilisant les loaders appropriés pour chaque type de fichier.
    Args:
        folder_path (str): Chemin du dossier à parcourir.
    Returns:
        List[Document]: Liste de documents chargés.
    """
    docs = []
    def aux_load_documents(folder, docs):
        """
        Fonction auxiliaire pour charger les documents d'un dossier.
        Args:
            folder (str): Chemin du dossier à parcourir.
            docs (List[Document]): Liste pour stocker les documents chargés.
        Returns:
            None
        """
        for file in Path(folder).glob("**/*"):
            if file.is_dir():
                aux_load_documents(str(file), docs)
            elif file.suffix == ".pdf":
                loader = PyPDFLoader(str(file))
                docs.extend(loader.load())
            elif file.suffix == ".md":
                loader = UnstructuredMarkdownLoader(str(file))
                docs.extend(loader.load())
            elif file.suffix == ".json":
                loader = JSONLoader(str(file), jq_schema='.', text_content=False)
                docs.extend(loader.load())
            else:
                continue
    aux_load_documents(folder_path, docs)
    print(f"{len(docs)} documents ont été chargés depuis {folder_path}")
    return docs

def clean_text(text: str) -> str:
    """
    Nettoie le texte en supprimant les espaces inutiles, les sauts de ligne excessifs,
    les en-têtes Markdown, et les références entre crochets.
    Args:
        text (str): Le texte à nettoyer.
    Returns:
        str: Le texte nettoyé.
    """
    # Normalisation Unicode
    text = unicodedata.normalize("NFKC", text)
    # Supprimer les emojis et symboles non alphanumériques
    text = re.sub(r"[^\w\s.,;:!?()\[\]\-’\"'éèàùâêîôûçÉÈÀÙÂÊÎÔÛÇ]", "", text)
    # Supprimer les en-têtes Markdown
    text = re.sub(r"^#+\s?", "", text, flags=re.MULTILINE)
    # Supprimer les mentions de type "Page X", "Chapitre X"
    # Supprimer la table des matières (naïvement via détection de titres + numéros de pages)
    lines = text.split('\n')
    lines = [line for line in lines if not re.match(r"^[\d\s]*[A-Z][A-Za-z\s]+\.{3,}\s*\d{1,3}$", line)]
    text = "\n".join(lines)
    # Supprimer la table des matières (naïvement via détection de titres + numéros de pages)
    lines = [line for line in lines if not re.match(r"^[\d\s]*[A-Z][A-Za-z\s]+\.{3,}\s*\d{1,3}$", line)]
    text = "\n".join(lines)
    return text

def lemmatize_text(text: str) -> str:
    """
    Applique la lemmatisation sur le texte.
    
    Args:
        text (str): Le texte à lemmatiser.
        
    Returns:
        str: Le texte lemmatisé.
    """
    text_doc = nlp(text)
    lemmatized = " ".join(
        [token.lemma_ for token in text_doc if not token.is_stop and not token.is_punct]
    )
    return lemmatized

def clean_documents(docs, lemmatize=False) -> List[Document]:
    """
    Prétraite les documents en nettoyant le texte.
    
    Args:
        docs (List[Document]): Liste de documents à prétraiter.
        
    Returns:
        List[Document]: Liste de documents avec le texte nettoyé.
    """
    for doc in docs:
        if doc.page_content:
            if lemmatize:
                doc.page_content = clean_text(lemmatize_text(doc.page_content))
            else:
                doc.page_content = clean_text(doc.page_content)
    return docs

def chunk_documents(documents, chunk_size, chunk_overlap):
    """
    Divise les documents en chunks de taille maximale fixe avec un chevauchement.
    
    Args:
        documents (List[Document]): Liste de documents à diviser.
        chunk_size (int): Taille maximale de chaque chunk.
        chunk_overlap (int): Nombre de caractères qui se chevauchent entre les chunks.
        
    Returns:
        List[Document]: Liste de documents divisés en chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n", "\n", ".", " "])
    return text_splitter.split_documents(documents)

def process_documents(folder_path, chunk_size=1000, chunk_overlap=200, lemmatize=False):
    """
    Charge et divise les documents d'un dossier en chunks.
    
    Args:
        folder_path (str): Chemin du dossier à parcourir.
        chunk_size (int): Taille maximale de chaque chunk.
        chunk_overlap (int): Nombre de caractères qui se chevauchent entre les chunks.
        
    Returns:
        List[Document]: Liste de documents divisés en chunks.
    """
    documents = []
    documents = load_documents(folder_path)
    documents = clean_documents(documents, lemmatize=lemmatize)
    
    if not documents:
        print("Aucun document trouvé.")
        return []
    
    print(f"{len(documents)} documents chargés. Division en chunks...")
    data = chunk_documents(documents, chunk_size, chunk_overlap)
    print(f"{len(data)} chunks créés.")
    return data

if __name__ == "__main__":
    folder_path = "data"  # Chemin du dossier contenant les documents
    chunk_size = 1000  # Taille maximale de chaque chunk
    chunk_overlap = 200  # Nombre de caractères qui se chevauchent entre les chunks
    
    # Traiter les documents et obtenir les chunks
    chunks = process_documents(folder_path, chunk_size, chunk_overlap)