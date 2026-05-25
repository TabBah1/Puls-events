import requests
import pandas as pd
import json
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

# Source officielle Open Agenda via OpenDataSoft - aucune clé requise
BASE_URL = "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/records"


def fetch_events_paris(max_events=500):
    """
    Récupère les événements Open Agenda filtrés sur Paris
    et sur les 12 derniers mois + événements à venir
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


def clean_events(raw_events):
    """
    Nettoie et structure les événements bruts
    du format Open Agenda / OpenDataSoft
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


def save_events(cleaned_events):
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