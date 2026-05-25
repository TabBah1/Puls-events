import json
import os
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")


def load_events():
    """
    Charge les événements nettoyés depuis le fichier JSON
    """
    with open("data/events_raw.json", "r", encoding="utf-8") as f:
        events = json.load(f)
    print(f"{len(events)} événements chargés")
    return events


def build_text_for_indexing(event):
    """
    Construit un texte riche pour chaque événement.
    C'est ce texte qui sera vectorisé.
    Plus il est informatif, meilleure sera la recherche.
    """
    parts = []

    if event.get("titre"):
        parts.append(f"Titre : {event['titre']}")

    if event.get("categories"):
        parts.append(f"Catégorie : {event['categories']}")

    if event.get("tags"):
        parts.append(f"Tags : {event['tags']}")

    if event.get("description"):
        parts.append(f"Description : {event['description']}")

    if event.get("description_longue"):
        parts.append(f"Détails : {event['description_longue']}")

    if event.get("lieu_nom"):
        parts.append(f"Lieu : {event['lieu_nom']}")

    if event.get("lieu_adresse"):
        parts.append(f"Adresse : {event['lieu_adresse']}")

    if event.get("lieu_ville"):
        parts.append(f"Ville : {event['lieu_ville']}")

    if event.get("date_debut"):
        # Formatage de la date pour la rendre lisible
        try:
            dt = datetime.fromisoformat(event["date_debut"])
            date_str = dt.strftime("%d/%m/%Y à %H:%M")
            parts.append(f"Date de début : {date_str}")
        except:
            parts.append(f"Date de début : {event['date_debut']}")

    if event.get("date_fin"):
        try:
            dt = datetime.fromisoformat(event["date_fin"])
            date_str = dt.strftime("%d/%m/%Y à %H:%M")
            parts.append(f"Date de fin : {date_str}")
        except:
            parts.append(f"Date de fin : {event['date_fin']}")

    if event.get("url"):
        parts.append(f"Plus d'infos : {event['url']}")

    return "\n".join(parts)


def create_documents(events):
    """
    Convertit chaque événement en objet Document LangChain
    avec son texte et ses métadonnées
    """
    documents = []

    for event in events:
        text = build_text_for_indexing(event)

        if not text.strip():
            continue

        # Les métadonnées sont stockées séparément du texte vectorisé
        # Elles permettront d'afficher les infos structurées dans les réponses
        metadata = {
            "uid": str(event.get("uid", "")),
            "titre": event.get("titre", ""),
            "lieu_ville": event.get("lieu_ville", "Paris"),
            "lieu_nom": event.get("lieu_nom", ""),
            "date_debut": event.get("date_debut", ""),
            "date_fin": event.get("date_fin", ""),
            "categories": event.get("categories", ""),
            "url": event.get("url", "")
        }

        documents.append(Document(page_content=text, metadata=metadata))

    print(f"{len(documents)} documents créés")
    return documents


def chunk_documents(documents):
    """
    Découpe les documents en chunks.
    chunk_size=500 : taille maximale de chaque morceau
    chunk_overlap=50 : chevauchement pour ne pas perdre le contexte
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "]
    )

    chunks = splitter.split_documents(documents)
    print(f"{len(chunks)} chunks créés après découpage")
    return chunks


def build_faiss_index(chunks):
    """
    Vectorise les chunks avec Mistral et construit l'index FAISS
    """
    print("Connexion à l'API Mistral et génération des embeddings...")
    print("Cette étape peut prendre quelques minutes selon le nombre de chunks...")

    embeddings = MistralAIEmbeddings(
        model="mistral-embed",
        mistral_api_key=MISTRAL_API_KEY
    )

    # Construction de l'index FAISS
    # LangChain envoie les chunks par batch à l'API Mistral
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Sauvegarde locale de l'index
    os.makedirs("data/faiss_index", exist_ok=True)
    vectorstore.save_local("data/faiss_index")

    print("Index FAISS sauvegardé dans data/faiss_index/")
    return vectorstore


if __name__ == "__main__":
    # 1. Chargement
    events = load_events()

    # 2. Création des documents LangChain
    documents = create_documents(events)

    # 3. Chunking
    chunks = chunk_documents(documents)

    # 4. Vectorisation + indexation FAISS
    vectorstore = build_faiss_index(chunks)

    # 5. Test rapide de recherche
    print("\n--- Test de recherche ---")
    query = "concert de musique à Paris ce weekend"
    results = vectorstore.similarity_search(query, k=3)

    for i, doc in enumerate(results):
        print(f"\nRésultat {i+1} : {doc.metadata.get('titre', 'Sans titre')}")
        print(f"  Lieu : {doc.metadata.get('lieu_nom', '')}")
        print(f"  Date : {doc.metadata.get('date_debut', '')}")