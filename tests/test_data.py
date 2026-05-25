import json
import pytest
from datetime import datetime, timedelta, timezone


EVENTS_FILE = "data/events_raw.json"
VILLE_CIBLE = "Paris"
DATE_LIMITE = datetime.now(timezone.utc) - timedelta(days=365)


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


# Test 4 : tous les événements sont géographiquement dans Paris
def test_evenements_localises_paris():
    events = load_events()
    for event in events:
        ville = event.get("lieu_ville", "")
        if ville:
            assert ville.lower() == VILLE_CIBLE.lower(), \
                f"Événement hors Paris détecté : {event.get('titre')} - Ville : {ville}"


# Test 5 : tous les événements datent de moins d'un an
def test_evenements_moins_un_an():
    events = load_events()
    for event in events:
        date_str = event.get("date_debut", "")
        if not date_str:
            continue
        try:
            date_event = datetime.fromisoformat(date_str)
            # Normalisation timezone
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