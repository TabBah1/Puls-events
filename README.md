# POC RAG — Puls-Events

Assistant de recommandation d'événements culturels parisiens basé sur un système RAG (Retrieval-Augmented Generation).

## Architecture

Open Agenda (OpenDataSoft) → Nettoyage → Chunking → Mistral Embeddings → FAISS → Chatbot

## Prérequis

- Python 3.11
- Anaconda
- Clé API Mistral (console.mistral.ai)
- Clé API Open Agenda (openagenda.com)

## Installation

```bash
conda create -n puls-events python=3.11 -y
conda activate puls-events
pip install -r requirements.txt
```

Créer un fichier `.env` à la racine :

MISTRAL_API_KEY=votre_clé_mistral
OPENAGENDA_API_KEY=votre_clé_openagenda

## Utilisation

### 1. Récupérer les données

```bash
python scripts/fetch_events.py
```

Récupère 500 événements culturels parisiens depuis Open Agenda (OpenDataSoft), filtrés sur les 12 derniers mois.

### 2. Construire la base vectorielle

```bash
python scripts/vectorize.py
```

Découpe les événements en chunks, génère les embeddings via Mistral et indexe dans FAISS.

### 3. Lancer le chatbot

```bash
python scripts/chatbot.py
```

### 4. Lancer les tests

```bash
pytest tests/test_data.py -v
```

### 5. Évaluer le système

```bash
python scripts/evaluate.py
```

Lance le chatbot sur les 8 questions annotées du jeu de test et sauvegarde les résultats dans `data/evaluation_results.json`.

## Structure du projet

```
puls-events-rag/
├── data/
│   ├── events_raw.json      # données nettoyées (généré)
│   ├── events.csv           # données au format CSV (généré)
│   ├── faiss_index/         # index vectoriel (généré)
│   ├── test_qa.json         # jeu de données annoté Q/R
│   └── evaluation_results.json  # résultats d'évaluation (généré)
├── scripts/
│   ├── fetch_events.py      # collecte et nettoyage des données
│   ├── vectorize.py         # chunking et vectorisation FAISS
│   ├── chatbot.py           # chatbot RAG interactif
│   └── evaluate.py          # évaluation sur jeu de données annoté
├── tests/
│   └── test_data.py         # 8 tests unitaires
├── .env                     # clés API (non versionné)
├── requirements.txt
└── README.md
```

## Choix techniques

| Composant | Choix | Justification |
|-----------|-------|---------------|
| LLM | mistral-small-latest | Bon rapport qualité/coût pour un POC |
| Embeddings | mistral-embed | Cohérence avec le LLM de génération |
| Vector store | FAISS CPU | Pas de serveur requis, suffisant pour un POC |
| Chunk size | 500 caractères | Équilibre précision sémantique / contexte |
| Chunk overlap | 50 caractères | Préserve le contexte entre les morceaux |
| Top-k retrieval | 5 documents | Contexte suffisant sans surcharger le prompt |

## Source de données

Les données proviennent du dataset public **Open Agenda** hébergé sur OpenDataSoft :
`https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/records`

Filtres appliqués : ville de Paris, événements de moins d'un an.
