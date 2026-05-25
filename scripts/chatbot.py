import os
from dotenv import load_dotenv
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
FAISS_INDEX_PATH = "data/faiss_index"


def load_vectorstore():
    """
    Charge l'index FAISS existant depuis le disque
    """
    embeddings = MistralAIEmbeddings(
        model="mistral-embed",
        mistral_api_key=MISTRAL_API_KEY
    )

    vectorstore = FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    print("Index FAISS chargé avec succès")
    return vectorstore


def build_rag_chain(vectorstore):
    """
    Construit la chaîne RAG :
    Question -> Recherche FAISS -> Prompt augmenté -> Mistral -> Réponse
    """

    # Retriever : récupère les 5 événements les plus pertinents
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # Prompt : instruit le LLM sur son rôle et le format attendu
    prompt_template = """
Tu es un assistant culturel spécialisé dans les événements à Paris.
Ton rôle est d'aider les utilisateurs à découvrir des événements culturels
en te basant UNIQUEMENT sur les informations fournies dans le contexte ci-dessous.

Règles importantes :
- Si l'information n'est pas dans le contexte, dis-le clairement
- Ne jamais inventer des dates, lieux ou événements
- Réponds en français, de manière naturelle et conviviale
- Si plusieurs événements correspondent, présente-les tous

CONTEXTE (événements disponibles) :
{context}

QUESTION :
{question}

RÉPONSE :
"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    # LLM Mistral pour la génération de réponses
    llm = ChatMistralAI(
        model="mistral-small-latest",
        mistral_api_key=MISTRAL_API_KEY,
        temperature=0.3
    )

    def format_docs(docs):
        """Formate les documents récupérés en texte pour le prompt"""
        formatted = []
        for i, doc in enumerate(docs):
            event_text = f"Événement {i+1}:\n{doc.page_content}"
            formatted.append(event_text)
        return "\n\n---\n\n".join(formatted)

    # Construction de la chaîne LangChain
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def run_chatbot():
    """
    Lance le chatbot en mode interactif
    """
    print("\n=== Chatbot Puls-Events - Événements culturels à Paris ===")
    print("Tapez 'quit' pour quitter\n")

    # Chargement de l'index
    vectorstore = load_vectorstore()

    # Construction de la chaîne RAG
    rag_chain = build_rag_chain(vectorstore)

    print("Chatbot prêt. Posez vos questions !\n")

    while True:
        question = input("Vous : ").strip()

        if not question:
            continue

        if question.lower() in ["quit", "exit", "quitter"]:
            print("Au revoir !")
            break

        print("\nAssistant : ", end="", flush=True)

        try:
            response = rag_chain.invoke(question)
            print(response)
        except Exception as e:
            print(f"Erreur : {e}")

        print()


if __name__ == "__main__":
    run_chatbot()