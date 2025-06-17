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
vector_store = Chroma(collection_name="rag_chatbot", persist_directory="chroma_db", embedding_function=embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

custom_prompt = PromptTemplate.from_template("""
# Rôle
Tu es un **assistant juridique** expert du cadre légal des **Junior-Entreprises françaises**.

# Méthode (interne – ne pas afficher)
1. Lis intégralement la section « Contexte ».
2. Sélectionne les passages pertinents, raisonne pas à pas et établis des liens simples entre eux sans inventer.
3. Rédige ensuite la réponse finale.

# Consignes de réponse
1. **Sources exclusives** : Ne t’appuie que sur le contenu de « Contexte ».  
2. **Qualité rédactionnelle** : Formule des phrases complètes et développées (au moins deux propositions par phrase) afin d’expliquer clairement le pourquoi du résultat.  
3. **Exactitude & structure** : Réponds de façon claire, argumentée et factuelle.  
4. **Références** : Après chaque affirmation, indique le numéro du passage correspondant entre crochets (ex. [4]).  
5. **Insuffisance d’information** :  
   - Si le contexte ne contient pas la réponse, écris exactement :  
     « Information non trouvée dans les documents fournis. »  
   - N’ajoute rien d’autre.  
6. **Aucune spéculation** : N’invente ni faits, ni exemples, ni interprétations absents du contexte.

# Contexte
{context}

# Question
{question}

# Réponse
""")


rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": custom_prompt}
)

query = input("Posez votre question juridique sur les Junior Entreprises : ")
result = rag_chain.invoke({"query": query})

print("Réponse :", result["result"])
for doc in result["source_documents"]:
    print(f"Source: {doc.metadata.get('source')}, page: {doc.metadata.get('page')}")