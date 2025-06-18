from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from dotenv import load_dotenv
load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2, google_api_key=os.getenv("GOOGLE_API_KEY"))

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector_store = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

custom_prompt = PromptTemplate.from_template("""
Tu es un assistant juridique spécialisé dans le cadre légal des Junior Entreprises françaises. Utilise les informations suivantes pour répondre à la question de manière factuelle et concise.

Contexte :
{context}

Question :
{question}

Réponse :
""")

rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": custom_prompt}
)

def process_documents():
    query = input("Posez votre question juridique sur les Junior Entreprises : ")
    result = rag_chain.invoke({"query": query})

    print("Réponse :", result["result"])
    for doc in result["source_documents"]:
        print(f"Source: {doc.metadata.get('source')}, page: {doc.metadata.get('page')}")
    
    
def answer(query: str) -> str :
    """
    Fonction pour répondre à une question juridique en utilisant RAG.
    
    Cette fonction utilise un modèle de langage pour générer une réponse basée sur des documents
    récupérés par un moteur de recherche vectoriel.
    
    Returns:
        str: La réponse générée par le modèle de langage.
    """
    result = rag_chain.invoke({"query": query})
    
    return result["result"]