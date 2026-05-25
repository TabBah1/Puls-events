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

    def format_docs(docs):
        return "\n\n---\n\n".join([doc.page_content for doc in docs])

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def evaluate():
    print("=== Évaluation du système RAG ===\n")

    with open(QA_FILE, "r", encoding="utf-8") as f:
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

    # Sauvegarde des résultats
    os.makedirs("data", exist_ok=True)
    with open("data/evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n=== Évaluation terminée ===")
    print(f"{len(results)} paires question/réponse évaluées")
    print("Résultats sauvegardés dans data/evaluation_results.json")
    print("\nNote : L'évaluation finale de la qualité est réalisée manuellement")
    print("en comparant les réponses générées aux réponses attendues.")


if __name__ == "__main__":
    evaluate()