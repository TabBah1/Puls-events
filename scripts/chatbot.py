"""
Module du chatbot RAG pour la recommandation d'événements culturels.

Ce module implémente la chaîne RAG complète :
    1. Chargement de l'index FAISS
    2. Retrieval des documents pertinents (top-k = 5)
    3. Construction du prompt augmenté
    4. Génération de la réponse via Mistral

Le chatbot fonctionne en mode interactif dans le terminal.
L'historique de conversation n'est pas conservé entre les échanges.

Auteur : Abdoulaye BAH
Date   : Mai 2026
"""

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


def load_vectorstore(index_path: str = FAISS_INDEX_PATH):
    """
    Charge l'index FAISS depuis le disque.

    Args:
        index_path (str): Chemin vers le dossier contenant index.faiss
                          et index.pkl. Par défaut FAISS_INDEX_PATH.

    Returns:
        FAISS: Instance du vectorstore LangChain prête à l'interrogation.

    Raises:
        FileNotFoundError: Si l'index FAISS est absent.
        Exception: En cas d'erreur d'initialisation du modèle d'embedding.

    Example:
        >>> vectorstore = load_vectorstore()
        >>> print("Index chargé")
    """
    embeddings = MistralAIEmbeddings(
        model="mistral-embed",
        mistral_api_key=MISTRAL_API_KEY
    )
    vectorstore = FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True
    )
    print("Index FAISS chargé avec succès")
    return vectorstore


def build_rag_chain(vectorstore):
    """
    Construit la chaîne RAG LangChain complète.

    La chaîne enchaîne :
        - Un retriever FAISS (top-k = 5 documents)
        - Un PromptTemplate avec instructions strictes anti-hallucination
        - Le LLM Mistral (mistral-small-latest, temperature=0.3)
        - Un parseur de sortie texte

    Args:
        vectorstore (FAISS): Instance du vectorstore LangChain chargé.

    Returns:
        Runnable: Chaîne LangChain invocable avec une question string.

    Note:
        Temperature 0.3 : favorise des réponses précises et stables
        plutôt que créatives, adapté aux recommandations d'événements.

    Example:
        >>> chain = build_rag_chain(vectorstore)
        >>> response = chain.invoke("Concerts de jazz à Paris ?")
        >>> print(response)
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    prompt_template = """
Tu es un assistant culturel spécialisé dans les événements à Paris.
Ton rôle est d'aider les utilisateurs à découvrir des événements culturels
en te basant UNIQUEMENT sur les informations fournies dans le contexte.

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

    llm = ChatMistralAI(
        model="mistral-small-latest",
        mistral_api_key=MISTRAL_API_KEY,
        temperature=0.3
    )

    def format_docs(docs: list) -> str:
        """
        Formate les documents récupérés en texte pour le prompt.

        Args:
            docs (list): Liste de documents LangChain récupérés par FAISS.

        Returns:
            str: Texte formaté avec séparateurs entre les événements.
        """
        formatted = []
        for i, doc in enumerate(docs):
            formatted.append(f"Événement {i+1}:\n{doc.page_content}")
        return "\n\n---\n\n".join(formatted)

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
    Lance le chatbot en mode interactif dans le terminal.

    Charge l'index FAISS, construit la chaîne RAG et entre dans
    une boucle de questions/réponses jusqu'à la saisie de 'quit'.

    Side effects:
        Affiche les réponses dans le terminal.
        Lit les questions depuis l'entrée standard.

    Example:
        >>> run_chatbot()
        === Chatbot Puls-Events ===
        Vous : Y a-t-il des concerts ce weekend ?
        Assistant : ...
    """
    print("\n=== Chatbot Puls-Events - Événements culturels à Paris ===")
    print("Tapez 'quit' pour quitter\n")

    vectorstore = load_vectorstore()
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