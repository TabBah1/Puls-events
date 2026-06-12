"""
Module de vectorisation et d'indexation des événements dans FAISS.

Ce module charge les événements nettoyés, construit des textes
d'indexation enrichis, découpe les textes en chunks, génère les
embeddings via l'API Mistral et construit l'index vectoriel FAISS
sauvegardé localement.

Le pipeline est reconstructible à tout moment en relançant ce script.

Auteur : Abdoulaye BAH
Date   : Mai 2026
"""

import json
import os
from datetime import datetime
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")


def load_events(filepath: str = "data/events_raw.json") -> list:
    """
    Charge les événements nettoyés depuis un fichier JSON.

    Args:
        filepath (str): Chemin vers le fichier JSON des événements.
                        Par défaut "data/events_raw.json".

    Returns:
        list: Liste de dictionnaires d'événements.

    Raises:
        FileNotFoundError: Si le fichier JSON est absent.
        json.JSONDecodeError: Si le fichier JSON est malformé.

    Example:
        >>> events = load_events()
        >>> print(len(events))
        497
    """
    with open(filepath, "r", encoding="utf-8") as f:
        events = json.load(f)
    print(f"{len(events)} événements chargés")
    return events


def build_text_for_indexing(event: dict) -> str:
    """
    Construit un texte riche et structuré à partir des métadonnées
    d'un événement.

    Ce texte sera vectorisé par le modèle d'embedding. Plus il est
    informatif, meilleure sera la précision de la recherche sémantique.

    Args:
        event (dict): Dictionnaire d'un événement nettoyé contenant
                      les champs titre, categories, description,
                      lieu_nom, lieu_adresse, lieu_ville, date_debut,
                      date_fin, url.

    Returns:
        str: Texte multi-lignes structuré prêt à la vectorisation.

    Example:
        >>> text = build_text_for_indexing(event)
        >>> print(text[:50])
        Titre : Festival de Jazz
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
        try:
            dt = datetime.fromisoformat(event["date_debut"])
            parts.append(f"Date de début : {dt.strftime('%d/%m/%Y à %H:%M')}")
        except Exception:
            parts.append(f"Date de début : {event['date_debut']}")
    if event.get("date_fin"):
        try:
            dt = datetime.fromisoformat(event["date_fin"])
            parts.append(f"Date de fin : {dt.strftime('%d/%m/%Y à %H:%M')}")
        except Exception:
            parts.append(f"Date de fin : {event['date_fin']}")
    if event.get("url"):
        parts.append(f"Plus d'infos : {event['url']}")

    return "\n".join(parts)


def create_documents(events: list) -> list:
    """
    Convertit les événements en objets Document LangChain.

    Chaque document contient le texte d'indexation enrichi et les
    métadonnées structurées de l'événement (titre, lieu, dates, url).

    Args:
        events (list): Liste de dictionnaires d'événements nettoyés.

    Returns:
        list: Liste d'objets Document LangChain prêts au chunking.

    Example:
        >>> docs = create_documents(events)
        >>> print(docs[0].metadata['titre'])
        Festival de Jazz
    """
    documents = []

    for event in events:
        text = build_text_for_indexing(event)
        if not text.strip():
            continue

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


def chunk_documents(documents: list) -> list:
    """
    Découpe les documents en chunks pour la vectorisation.

    Utilise RecursiveCharacterTextSplitter avec une taille de 500
    caractères et un chevauchement de 50 caractères pour préserver
    le contexte entre les morceaux.

    Args:
        documents (list): Liste d'objets Document LangChain.

    Returns:
        list: Liste de chunks (objets Document) prêts à la vectorisation.

    Note:
        Le chevauchement (overlap) évite de couper les informations
        cruciales comme les dates ou adresses en plein milieu.

    Example:
        >>> chunks = chunk_documents(documents)
        >>> print(len(chunks))
        2025
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(documents)
    print(f"{len(chunks)} chunks créés après découpage")
    return chunks


def build_faiss_index(chunks: list, index_path: str = "data/faiss_index"):
    """
    Vectorise les chunks et construit l'index FAISS local.

    Appelle l'API Mistral pour générer les embeddings de chaque chunk,
    construit l'index FAISS en mémoire, puis le sauvegarde sur disque
    pour réutilisation sans recalcul.

    Args:
        chunks (list): Liste de chunks (objets Document LangChain).
        index_path (str): Chemin du dossier de sauvegarde de l'index.
                          Par défaut "data/faiss_index".

    Returns:
        FAISS: Instance du vectorstore FAISS chargée en mémoire.

    Side effects:
        Crée ou écrase le dossier index_path avec index.faiss et index.pkl.

    Raises:
        Exception: En cas d'erreur d'appel à l'API Mistral.

    Example:
        >>> vectorstore = build_faiss_index(chunks)
        >>> results = vectorstore.similarity_search("concert jazz", k=3)
    """
    print("Connexion à l'API Mistral et génération des embeddings...")
    print("Cette étape peut prendre quelques minutes...")

    embeddings = MistralAIEmbeddings(
        model="mistral-embed",
        mistral_api_key=MISTRAL_API_KEY
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    os.makedirs(index_path, exist_ok=True)
    vectorstore.save_local(index_path)

    print(f"Index FAISS sauvegardé dans {index_path}/")
    return vectorstore


if __name__ == "__main__":
    events = load_events()
    documents = create_documents(events)
    chunks = chunk_documents(documents)
    vectorstore = build_faiss_index(chunks)

    print("\n--- Test de recherche ---")
    query = "concert de musique à Paris ce weekend"
    results = vectorstore.similarity_search(query, k=3)

    for i, doc in enumerate(results):
        print(f"\nRésultat {i+1} : {doc.metadata.get('titre', 'Sans titre')}")
        print(f"  Lieu : {doc.metadata.get('lieu_nom', '')}")
        print(f"  Date : {doc.metadata.get('date_debut', '')}")