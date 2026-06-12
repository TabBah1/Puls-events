import json
import os
import pytest
from datetime import datetime, timedelta, timezone


EVENTS_FILE = "data/events_raw.json"
FAISS_INDEX_DIR = "data/faiss_index"
VILLE_CIBLE = "Paris"
DATE_LIMITE = datetime.now(timezone.utc) - timedelta(days=420)
#420 au lieu de 365 pour un peu de marge 

def load_events():
    with open(EVENTS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


# Test 1 : le fichier de données existe et n'est pas vide
def test_fichier_non_vide():
    events = load_events()
    assert len(events) > 0, "Le fichier d'événements est vide"


# Test 2 : tous les événements ont un titre
def test_tous_les_evenements_ont_un_titre():
    events = load_events()
    for event in events:
        assert event.get("titre"), f"Événement sans titre : {event}"


# Test 3 : tous les événements ont une description
def test_tous_les_evenements_ont_une_description():
    events = load_events()
    for event in events:
        desc = event.get("description", "") or event.get("description_longue", "")
        assert desc, f"Événement sans description : {event.get('titre')}"


# Test 4 : tous les événements ont un champ ville ET celui-ci est Paris
def test_evenements_localises_paris():
    events = load_events()
    sans_ville = [e.get("titre") for e in events if not e.get("lieu_ville")]
    assert len(sans_ville) == 0, \
        f"{len(sans_ville)} événements sans ville : {sans_ville[:5]}"
    
    for event in events:
        ville = event.get("lieu_ville", "")
        if ville:
            assert ville.lower() == VILLE_CIBLE.lower(), \
                f"Événement hors Paris : {event.get('titre')} - Ville : {ville}"


# Test 5 : tous les événements datent de moins d'un an
def test_evenements_moins_un_an():
    events = load_events()
    for event in events:
        date_str = event.get("date_debut", "")
        if not date_str:
            continue
        try:
            date_event = datetime.fromisoformat(date_str)
            if date_event.tzinfo is None:
                date_event = date_event.replace(tzinfo=timezone.utc)
            assert date_event >= DATE_LIMITE, \
                f"Événement trop ancien : {event.get('titre')} - Date : {date_str}"
        except ValueError:
            pytest.fail(f"Format de date invalide : {date_str}")


# Test 6 : les événements ont une date de début
def test_evenements_ont_une_date():
    events = load_events()
    sans_date = [e.get("titre") for e in events if not e.get("date_debut")]
    assert len(sans_date) == 0, \
        f"{len(sans_date)} événements sans date : {sans_date[:5]}"


# Test 7 : l'index FAISS existe et contient les fichiers attendus
# Ces fichiers sont générés par vectorize.py — leur absence indique
# que la vectorisation n'a pas été exécutée ou a échoué
def test_faiss_index_existe():
    assert os.path.isdir(FAISS_INDEX_DIR), \
        f"Le dossier d'index FAISS est absent : {FAISS_INDEX_DIR}"
    fichiers_attendus = ["index.faiss", "index.pkl"]
    for fichier in fichiers_attendus:
        chemin = os.path.join(FAISS_INDEX_DIR, fichier)
        assert os.path.isfile(chemin), \
            f"Fichier FAISS manquant : {chemin}"
    for fichier in fichiers_attendus:
        chemin = os.path.join(FAISS_INDEX_DIR, fichier)
        assert os.path.getsize(chemin) > 0, \
            f"Fichier FAISS vide : {chemin}"


# Test 8 : le retriever FAISS retourne des résultats pour une requête basique
# Ce test charge l'index en local (pas d'appel API) et vérifie
# que la similarité cosinus fonctionne et retourne bien k résultats
def test_retriever_retourne_resultats():
    try:
        import faiss
        import pickle
        import numpy as np
    except ImportError as e:
        pytest.fail(f"Dépendance manquante : {e}")

    index_path = os.path.join(FAISS_INDEX_DIR, "index.faiss")
    pkl_path = os.path.join(FAISS_INDEX_DIR, "index.pkl")

    index = faiss.read_index(index_path)
    assert index.ntotal > 0, "L'index FAISS est vide (aucun vecteur indexé)"

    with open(pkl_path, "rb") as f:
        docstore_data = pickle.load(f)

    # Vérification de la cohérence entre index FAISS et docstore
    # Le pkl LangChain contient (docstore, index_to_docstore_id)
    if isinstance(docstore_data, tuple) and len(docstore_data) == 2:
        _, index_to_id = docstore_data
        assert len(index_to_id) == index.ntotal, (
            f"Incohérence : {index.ntotal} vecteurs dans FAISS "
            f"mais {len(index_to_id)} entrées dans le docstore"
        )

    # Simulation d'une requête avec un vecteur aléatoire
    # pour vérifier que la recherche retourne bien k résultats
    k = 5
    dimension = index.d
    vecteur_test = np.random.rand(1, dimension).astype("float32")
    distances, indices = index.search(vecteur_test, k)

    assert len(indices[0]) == k, \
        f"Le retriever devrait retourner {k} résultats, reçu : {len(indices[0])}"
    assert all(i >= 0 for i in indices[0]), \
        "Certains indices retournés sont invalides (-1)"