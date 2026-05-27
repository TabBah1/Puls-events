"""
Module de collecte et de nettoyage des données événementielles.

Ce module récupère les événements culturels parisiens depuis le dataset
public Open Agenda hébergé sur OpenDataSoft, applique des filtres
géographiques et temporels, puis sauvegarde les données nettoyées
pour la vectorisation.

Source : https://public.opendatasoft.com/api/explore/v2.1/catalog/
         datasets/evenements-publics-openagenda/records

Auteur : Ingénieur Data Freelance — Puls-Events
Date   : Mai 2026
"""

import requests
import pandas as pd
import json
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

BASE_URL = (
    "https://public.opendatasoft.com/api/explore/v2.1/"
    "catalog/datasets/evenements-publics-openagenda/records"
)


def fetch_events_paris(max_events: int = 500) -> list:
    """
    Récupère les événements culturels parisiens via l'API OpenDataSoft.

    Applique un filtre temporel sur les 12 derniers mois et un filtre
    géographique sur la ville de Paris. La pagination est gérée
    automatiquement par offset.

    Args:
        max_events (int): Nombre maximum d'événements à récupérer.
                          Par défaut 500.

    Returns:
        list: Liste de dictionnaires bruts au format OpenDataSoft.
              Retourne une liste vide en cas d'erreur API.

    Raises:
        requests.exceptions.RequestException: En cas de problème réseau.

    Example:
        >>> events = fetch_events_paris(max_events=100)
        >>> print(len(events))
        100
    """
    date_from = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

    all_events = []
    offset = 0
    limit = 100

    print("Récupération des événements Open Agenda - Paris...")

    while len(all_events) < max_events:
        params = {
            "limit": limit,
            "offset": offset,
            "where": f'location_city="Paris" AND firstdate_begin >= "{date_from}"',
            "lang": "fr"
        }

        response = requests.get(BASE_URL, params=params)

        if response.status_code != 200:
            print(f"Erreur API {response.status_code}: {response.text}")
            break

        data = response.json()
        results = data.get("results", [])

        if not results:
            print("Plus d'événements disponibles.")
            break

        all_events.extend(results)
        print(f"  {len(all_events)} événements récupérés...")

        total = data.get("total_count", 0)
        if len(all_events) >= total or len(all_events) >= max_events:
            break

        offset += limit

    print(f"\nTotal récupéré : {len(all_events)} événements")
    return all_events


def clean_events(raw_events: list) -> list:
    """
    Nettoie et structure les événements bruts issus de l'API OpenDataSoft.

    Extrait les champs pertinents pour le système RAG, construit
    les textes d'indexation et filtre les entrées invalides (sans titre
    ni description).

    Args:
        raw_events (list): Liste de dictionnaires bruts retournés
                           par l'API OpenDataSoft.

    Returns:
        list: Liste de dictionnaires nettoyés et structurés contenant
              les champs : uid, titre, description, description_longue,
              lieu_nom, lieu_ville, lieu_adresse, date_debut, date_fin,
              categories, tags, url.

    Example:
        >>> raw = fetch_events_paris(100)
        >>> cleaned = clean_events(raw)
        >>> print(cleaned[0].keys())
        dict_keys(['uid', 'titre', 'description', ...])
    """
    cleaned = []

    for event in raw_events:
        try:
            title = event.get("title_fr", event.get("title_en", "Sans titre"))
            description = event.get("description_fr", event.get("description_en", ""))
            long_desc = event.get("longdescription_fr", event.get("longdescription_en", ""))
            desc_finale = description if description else long_desc

            lieu_nom = event.get("location_name", "")
            lieu_ville = event.get("location_city", "Paris")
            lieu_adresse = event.get("location_address", "")
            date_debut = event.get("firstdate_begin", "")
            date_fin = event.get("lastdate_end", "")
            categories = event.get("category_fr", "")
            tags = event.get("tags_fr", "")
            url = event.get("canonicalurl", "")

            if not title or not desc_finale:
                continue

            cleaned.append({
                "uid": event.get("uid", ""),
                "titre": title,
                "description": description,
                "description_longue": long_desc,
                "lieu_nom": lieu_nom,
                "lieu_ville": lieu_ville,
                "lieu_adresse": lieu_adresse,
                "date_debut": date_debut,
                "date_fin": date_fin,
                "categories": categories,
                "tags": tags,
                "url": url
            })

        except Exception as e:
            print(f"Erreur sur un événement : {e}")
            continue

    print(f"Événements valides après nettoyage : {len(cleaned)}")
    return cleaned


def save_events(cleaned_events: list) -> pd.DataFrame:
    """
    Sauvegarde les événements nettoyés en JSON et CSV.

    Crée le dossier data/ si absent. Les deux formats sont générés
    pour faciliter l'inspection manuelle (CSV) et la vectorisation (JSON).

    Args:
        cleaned_events (list): Liste de dictionnaires nettoyés produits
                                par clean_events().

    Returns:
        pd.DataFrame: DataFrame pandas des événements sauvegardés.

    Side effects:
        Crée ou écrase data/events_raw.json et data/events.csv.

    Example:
        >>> df = save_events(cleaned)
        >>> print(df.shape)
        (497, 11)
    """
    os.makedirs("data", exist_ok=True)

    with open("data/events_raw.json", "w", encoding="utf-8") as f:
        json.dump(cleaned_events, f, ensure_ascii=False, indent=2)

    df = pd.DataFrame(cleaned_events)
    df.to_csv("data/events.csv", index=False, encoding="utf-8-sig")

    print(f"Données sauvegardées dans data/events.csv et data/events_raw.json")
    return df


if __name__ == "__main__":
    raw_events = fetch_events_paris(max_events=500)

    if not raw_events:
        print("Aucun événement récupéré.")
        exit()

    cleaned = clean_events(raw_events)
    df = save_events(cleaned)

    print("\n--- Aperçu des données ---")
    print(df[["titre", "lieu_ville", "date_debut", "categories"]].head(10))