"""
Module d'évaluation du système RAG sur un jeu de données annoté.

Ce module charge les paires questions/réponses annotées manuellement,
interroge la chaîne RAG sur chaque question et sauvegarde les réponses
générées pour comparaison avec les réponses de référence.

Le jeu de données annoté (data/test_qa.json) constitue le Ground Truth
permettant d'évaluer la qualité et la pertinence des réponses du chatbot.

Auteur : Abdoulaye BAH
Date   : Mai 2026
"""

import json
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
QA_FILE = "data/test_qa.json"


def load_rag_chain():
    """
    Initialise et retourne la chaîne RAG complète.

    Charge l'index FAISS depuis le disque, configure le retriever
    (k=5), le prompt anti-hallucination et le LLM Mistral.

    Returns:
        Runnable: Chaîne LangChain invocable avec une question string.

    Raises:
        FileNotFoundError: Si l'index FAISS est absent.

    Example:
        >>> chain = load_rag_chain()
        >>> response = chain.invoke("Concerts jazz Paris ?")
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
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    prompt_template = """
Tu es un assistant culturel spécialisé dans les événements à Paris.
Réponds en te basant UNIQUEMENT sur le contexte fourni.
Si l'information n'est pas dans le contexte, dis-le clairement.

CONTEXTE :
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
            docs (list): Documents LangChain récupérés par FAISS.

        Returns:
            str: Texte formaté avec séparateurs.
        """
        return "\n\n---\n\n".join([doc.page_content for doc in docs])

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


def evaluate(qa_file: str = QA_FILE, output_file: str = "data/evaluation_results.json"):
    """
    Évalue le système RAG sur le jeu de données annoté.

    Pour chaque paire question/réponse du fichier annoté, interroge
    la chaîne RAG et sauvegarde la réponse générée pour comparaison
    avec la réponse de référence humaine.

    Args:
        qa_file (str): Chemin vers le fichier JSON du jeu annoté.
                       Par défaut "data/test_qa.json".
        output_file (str): Chemin de sauvegarde des résultats.
                           Par défaut "data/evaluation_results.json".

    Returns:
        list: Liste de dictionnaires contenant question,
              reponse_attendue et reponse_generee.

    Side effects:
        Crée ou écrase output_file avec les résultats d'évaluation.

    Note:
        L'évaluation finale de la qualité est réalisée manuellement
        en comparant les réponses générées aux réponses attendues.
        Pour une évaluation automatique, utiliser RAGAS.

    Example:
        >>> results = evaluate()
        >>> print(len(results))
        8
    """
    print("=== Évaluation du système RAG ===\n")

    with open(qa_file, "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)

    chain = load_rag_chain()
    results = []

    for i, pair in enumerate(qa_pairs):
        question = pair["question"]
        reponse_attendue = pair["reponse_attendue"]

        print(f"Question {i+1}/{len(qa_pairs)} : {question[:60]}...")
        reponse_generee = chain.invoke(question)

        results.append({
            "question": question,
            "reponse_attendue": reponse_attendue,
            "reponse_generee": reponse_generee
        })
        print(f"  Réponse générée : {reponse_generee[:100]}...\n")

    os.makedirs("data", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"=== Évaluation terminée ===")
    print(f"{len(results)} paires évaluées")
    print(f"Résultats sauvegardés dans {output_file}")
    return results


if __name__ == "__main__":
    evaluate()