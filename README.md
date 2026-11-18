# DocuAgent – AI-Powered Document Assistant

**Status: 🚧 Work in Progress | Active Development**

DocuAgent is an AI-powered document assistant built with Python, FastAPI, Streamlit, and Retrieval-Augmented Generation (RAG).

The application allows users to upload multiple PDF documents, ask questions in natural language, and receive AI-generated answers with references to the original documents.

This project is being developed as a practical portfolio project focused on **Generative AI, semantic search, document processing, and AI application development**.

## Current Features

- **Multi-PDF Upload:** Upload and index multiple PDF documents.
- **PDF Text Extraction:** Extract text and preserve page information.
- **Token-Based Chunking:** Split documents into manageable text sections with overlap.
- **Semantic Search:** Find relevant document sections using multilingual sentence embeddings and cosine similarity.
- **Retrieval-Augmented Generation (RAG):** Generate answers using OpenAI GPT-4.1-mini and retrieved document context.
- **Source References:** Display document names, page numbers, and chunk references.
- **Interactive Interface:** Ask questions through a Streamlit chat application.
- **Document Management:** List and remove indexed documents.
- **Automated Tests:** Unit and API tests for core functionality.

## Technology Stack

| Category             | Technologies          |
| -------------------- | --------------------- |
| Programming Language | Python                |
| Backend              | FastAPI               |
| Frontend             | Streamlit             |
| AI / LLM             | OpenAI GPT-4.1-mini   |
| Embeddings           | Sentence Transformers |
| Semantic Search      | Cosine Similarity     |
| PDF Processing       | PyMuPDF               |
| Testing              | pytest                |
| Version Control      | Git & GitHub          |

## How It Works

1. **Document Upload:** Users upload PDF files through Streamlit.
2. **Text Extraction:** The backend extracts text and page information.
3. **Chunking:** Text is divided into token-based sections.
4. **Embedding Generation:** A local Sentence Transformers model converts text chunks into numerical vectors.
5. **Semantic Retrieval:** The system identifies relevant chunks based on the user's question.
6. **Answer Generation:** GPT-4.1-mini generates an answer using the retrieved context.
7. **Source Attribution:** The application displays references to the original documents.

## Current Architecture

```text
                  Streamlit UI
                       |
                  FastAPI Backend
                       |
              PDF Document Processing
                       |
               Token-Based Chunking
                       |
              Local Embedding Model
                       |
                Semantic Search
                       |
                 Relevant Chunks
                       |
                OpenAI GPT-4.1-mini
                       |
              Answer + PDF Sources
```

## Development Roadmap

- [x] FastAPI backend setup
- [x] PDF text extraction
- [x] Token-based text chunking
- [x] Local embedding generation
- [x] Semantic search
- [x] RAG answer generation
- [x] Streamlit chat interface
- [x] Multi-document support
- [x] Document and page references
- [x] Automated testing
- [x] Persistent document and embedding storage with SQLite
- [ ] Advanced retrieval and answer-quality evaluation
- [ ] AI agent workflows with LangGraph
- [ ] Improved document analysis and information extraction
- [ ] Deployment and CI/CD

## Current Limitations

DocuAgent is currently a local development prototype.

- Indexed document metadata, chunks, and embeddings are persisted locally in SQLite and restored on backend startup. Original PDF files are not stored.
- The application is designed for local single-user use.
- OpenAI API access is required for answer generation.
- Source references do not guarantee that every generated answer is factually correct.
- Production authentication and access controls have not yet been implemented.

## Getting Started

### Retrieval-Evaluation

Der Datensatz [retrieval_questions.json](data/evaluation/retrieval_questions.json)
enthält 20 Fragen mit überprüfbaren Textbelegen aus den beiden künstlichen PDFs.
Die Evaluation verändert weder die RAG-Pipeline noch die SQLite-Dokumente.
Sie verwendet nur das vorhandene lokale Embedding-Modell und keine OpenAI API.

Aus `backend` ausführen:

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_retrieval.py
.\.venv\Scripts\python.exe -m pytest -v
```

Der tatsächlich ausgeführte lokale Modelllauf ergab **Hit Rate@1: 75 %**,
**Hit Rate@3: 95 %** und **MRR: 0,8625** bei 20 Fragen und 18 Chunks.
Die vollständigen Fragen, erwarteten Quellen, Ränge und Scores stehen im
[Evaluationsbericht](docs/retrieval_evaluation.md); der
[JSON-Bericht](docs/retrieval_evaluation.json) enthält zusätzlich alle Rangfolgen
und Fingerabdrücke der Eingabedateien. Ein erneuter Lauf überschreibt diese Berichte.
Automatisierte Validierung: **182 Tests bestanden** (`python -m pytest -v`).

Hit Rate@1/3 misst, wie oft ein belegender Chunk unter den ersten 1/3 Treffern liegt.
MRR mittelt den Kehrwert des ersten relevanten Rangs: Rang 1 ergibt 1, Rang 2 ergibt
0,5, ein fehlender Treffer 0. Die Auswertung nutzt die vollständige Rangliste.
Dateiname, Seite und Textbelege bestimmen die Relevanz; Beispielfragen ohne Antwort
zählen nicht. Alternative gültige Quellen sind zugelassen. Der kleine Datensatz
misst ausschließlich Retrieval auf diesen Testdokumenten, keine LLM-Antwortqualität.

### Hybrid Retrieval mit BM25 und RRF

Die semantische Suche bleibt der Standard. Optional kombiniert `hybrid` sie mit
`BM25Okapi` aus dem etablierten Paket [rank-bm25](https://github.com/dorianbrown/rank_bm25).
BM25 bewertet gemeinsame Wörter anhand von Häufigkeit, Seltenheit und Chunklänge.
Dokumente und Fragen werden identisch in Unicode-Wörter/Zahlen zerlegt und mit
`casefold` normalisiert. Es gibt derzeit kein Stemming und keine Stoppwortliste.

Reciprocal Rank Fusion kombiniert die Ranglisten mit
`RRF = 1/(60 + Rang_semantic) + 1/(60 + Rang_bm25)`.
Fehlende BM25-Treffer tragen 0 bei. Beide Verfahren sind gleich gewichtet.
Die Konstante 60 folgt der [RRF-Originalarbeit](https://cormack.uwaterloo.ca/cormack/cormacksigir09-rrf.pdf).
BM25 verwendet k1=1,5, b=0,75 und epsilon=0,25. Diese Parameter wurden vor der
Evaluation festgelegt und nicht an die 20 Fragen angepasst.
Beide Verfahren berücksichtigen alle vorhandenen Chunks; BM25 liefert nur Chunks
mit einer Wortübereinstimmung. Bei Gleichstand bleibt die Reihenfolge stabil.

Im PowerShell-Terminal des Backends vor dem Start auswählen, ohne `.env` zu ändern:

```powershell
$env:DOCUAGENT_RETRIEVAL_MODE = "hybrid"  # alternativ: "semantic" (Standard)
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Die API-Anfragen und Streamlit bleiben unverändert. Im Python-Service kann der
Modus pro Suche explizit gewählt werden: `collection.search(query, mode="semantic")`
oder `collection.search(query, mode="hybrid")`. Dokument-IDs und Chunk-Metadaten
bleiben erhalten. `similarity_score` ist weiterhin die unveränderte
Kosinus-Ähnlichkeit; Hybrid sortiert nach dem zusätzlichen `rrf_score` und liefert
auch `bm25_score`, `semantic_rank` und `bm25_rank`. Die Scores haben unterschiedliche
Skalen. SQLite speichert weiterhin die vorhandenen Dokumente und Embeddings.
Der BM25-Index wird bei Bedarf aus diesen Chunks aufgebaut, wiederverwendet und bei
Hinzufügen/Löschen eines Dokuments neu erstellt. Dokument-Embeddings werden dafür
nicht erneut berechnet.

Vergleich mit exakt demselben Datensatz und einmalig vorbereiteten Embeddings:

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_retrieval.py --mode compare
# Einzeln weiterhin möglich:
.\.venv\Scripts\python.exe scripts/evaluate_retrieval.py --mode semantic
.\.venv\Scripts\python.exe scripts/evaluate_retrieval.py --mode hybrid
```

`compare` ist der Standard des Evaluationsskripts. Die tatsächlichen Ergebnisse:

| Methode | Hit Rate@1 | Hit Rate@3 | MRR |
| --- | ---: | ---: | ---: |
| Semantic | 75 % | 95 % | 0,862500 |
| Hybrid | 65 % | 80 % | 0,765833 |

Bei 20 Fragen verbessern sich 1, verschlechtern sich 4 und bleiben 15 unverändert.
Sicherheitsupdates steigen von Rang 2 auf 1; Urlaubstage fallen von 1 auf 2,
Urlaubsantrag und Rechnungsfrist von 1 auf 5, Passwortlänge von 2 auf 6.
Die Keyword-Suche erkennt Wortformen wie „Passwort/Passwörter“ nicht als identisch.
Hybrid ist in diesem Test schlechter; deshalb bleibt `semantic` der Standard.
Es wurden weder Rankings nachbearbeitet noch die Parameter nachträglich optimiert.
Alle 20 Rangvergleiche stehen im [Vergleichsbericht](docs/hybrid_retrieval_comparison.md),
die vollständigen Rohscores im [JSON-Bericht](docs/hybrid_retrieval_comparison.json).
Es erfolgen keine LLM-Aufrufe und keine Änderungen an `.env` oder der Nutzerdatenbank.
Validierung nach der Hybrid-Erweiterung: **201 Tests bestanden**; `pip check`
meldet keine Abhängigkeitsprobleme.

### Deutsche BM25-Wortnormalisierung

Der zusätzliche Modus `hybrid_de` verwendet den deutschen
[Snowball-Stemmer](https://snowballstem.org/algorithms/german/stemmer.html)
aus `snowballstemmer` (lokal installiert: Version 3.1.1). Dokument-Chunks und
Suchfragen werden identisch mit Unicode-NFC, `casefold`, Wortzerlegung und
Snowball-Stemming normalisiert. Texte, Embeddings und Quellen bleiben unverändert.
Der bisherige Modus `hybrid` bleibt ohne Stemming verfügbar. `semantic` bleibt Standard.

```powershell
# Im Backend-Terminal optional auswählen; keine Änderung an .env:
$env:DOCUAGENT_RETRIEVAL_MODE = "hybrid_de"
# Vergleich aller drei Varianten aus backend:
.\.venv\Scripts\python.exe scripts/evaluate_retrieval.py --mode compare
```

Der neue Vergleich verwendet exakt dieselben 20 Fragen, Referenzen und 18 Chunks.
Dokument-Embeddings werden für alle drei Varianten nur einmal berechnet. RRF bleibt
gleichgewichtet mit Konstante 60; auch k1=1,5, b=0,75 und epsilon=0,25 bleiben
unverändert. Keine Stoppwörter, Synonymregeln oder testspezifischen Wörterbücher wurden ergänzt.

| Methode | Hit Rate@1 | Hit Rate@3 | MRR |
| --- | ---: | ---: | ---: |
| Semantic | 75 % | 95 % | 0,862500 |
| Hybrid bisher | 65 % | 80 % | 0,765833 |
| Hybrid mit Snowball DE | 60 % | 95 % | 0,776667 |

Die vier bisherigen Verschlechterungen lassen sich anhand der tatsächlichen
BM25-Ränge erklären:

- **Urlaubstage:** Der relevante Chunk hat nur die Worttreffer `urlaubstage` und
  `pro`, BM25-Rang 3. Beispielfragen rangieren höher. Stemming macht daraus
  `urlaubstag` und `pro`, verbessert den Rang aber nicht: Hybrid bleibt Rang 2.
- **Urlaubsantrag:** Bisher keine gemeinsamen Wörter im relevanten Chunk. Snowball
  gleicht unter anderem `Urlaubsantrag/Urlaubsanträge` und `muss/müssen` an.
  Der Chunk erhält BM25-Rang 3 und steigt in der Fusion von 5 auf 2.
  `einreichen/eingereicht` bleiben jedoch verschiedene Stämme.
- **Lieferantenrechnungen:** Bisher trifft im relevanten Chunk nur `werden` zu,
  BM25-Rang 13. Nach Stemming treffen `muss` und `werd` zu, BM25-Rang 12.
  `Lieferantenrechnungen/Rechnungen` werden nicht als identischer Begriff erkannt:
  Snowball zerlegt keine zusammengesetzten Wörter. Die Fusion bleibt Rang 5.
- **Passwortlänge:** Bisher kein Worttreffer. `Passwort/Passwörter` und
  `muss/müssen` werden angeglichen. BM25-Rang 3; Fusion verbessert sich von 6 auf 2.

Zusätzlich verbessert sich die Meldefrist für verlorene Firmenlaptops von Rang 4
auf 3. Bei der Einkaufsfreigabe entsteht dagegen eine neue Verschlechterung von
Rang 1 auf 2: Der relevante Chunk fällt in BM25 von Rang 2 auf 3. Die neuen Stämme
verändern auch Dokumenthäufigkeiten, IDF-Werte und Treffer konkurrierender Chunks;
ein Beispielfragen-Chunk erhält den ersten Platz. Stemming ist daher keine Garantie
für eine bessere Rangfolge.

Gegenüber bisherigem Hybrid: **3 Fragen besser, 1 schlechter, 16 unverändert**.
Gegenüber Semantic: **2 besser, 4 schlechter, 14 unverändert**. Alle Fragen mit
Rängen, Verbesserungen/Verschlechterungen und Token-Diagnosen stehen im
[neuen Vergleichsbericht](docs/german_hybrid_retrieval_comparison.md) und in den
[vollständigen JSON-Rohdaten](docs/german_hybrid_retrieval_comparison.json).
Die bisherigen Zweier-Vergleichsberichte bleiben als historische Baseline erhalten.
`--mode compare` erzeugt jetzt standardmäßig den neuen Dreier-Vergleich;
`--mode semantic`, `--mode hybrid` und `--mode hybrid_de` sind einzeln nutzbar.

Der kleine Datensatz zeigt weiterhin keinen Vorteil gegenüber Semantic Search.
Jede Verbesserung muss mit unabhängigen Fragen bestätigt werden, bevor eine
Standardumstellung gerechtfertigt ist. Es gab keine nachträgliche Optimierung
der RRF-Gewichte, keine LLM-Aufrufe und keine Änderungen an `.env` oder SQLite.
Validierung: **212 Tests bestanden**, `pip check` ohne Abhängigkeitsprobleme.

### Lokale SQLite-Speicherung testen

Die Datenbank wird beim FastAPI-Start automatisch unter
`data/storage/docuagent.sqlite3` angelegt. Der Pfad ist unabhängig vom
Arbeitsverzeichnis. Optional kann `DOCUAGENT_DB_PATH` einen anderen lokalen Pfad
festlegen; relative Pfade beziehen sich auf den Projektordner. Die Standarddatenbank
und ihre Journaldateien sind von Git ausgeschlossen.

1. FastAPI und Streamlit wie bisher mit dem Interpreter aus `backend/.venv` starten.
2. Künstliche PDFs über Streamlit hinzufügen und die Dokumentliste prüfen.
3. Nur FastAPI mit `Strg+C` stoppen und mit demselben Befehl neu starten.
4. In Streamlit „Dokumentliste aktualisieren“ wählen. Dokumente und IDs bleiben erhalten;
   eine erneute Indexierung ist nicht nötig.
5. Eine Suche über `POST /documents/search` testen. Sie verwendet gespeicherte
   Dokument-Embeddings und benötigt keinen OpenAI-Aufruf. Chatfragen über
   `/documents/ask` verwenden weiterhin die kostenpflichtige OpenAI API.
6. Ein Dokument entfernen und FastAPI erneut starten: Es bleibt gelöscht.

SQLite speichert Dokumenttexte lokal im Klartext. Bereits vor dieser Änderung nur
im Arbeitsspeicher vorhandene Dokumente müssen einmal neu indexiert werden.
Der Prototyp ist weiterhin für einen lokalen Benutzer und einen FastAPI-Prozess
vorgesehen. Der bisherige Endpunkt `/documents/index` ersetzt auch in SQLite den
gesamten Index; die Streamlit-Oberfläche verwendet zum Hinzufügen `/documents`.

A complete installation and configuration guide will be added as development progresses.

The project currently runs locally using FastAPI and Streamlit. An OpenAI API key is required for AI-generated answers and must be configured through environment variables.

**Never commit API keys or `.env` files to GitHub.**

## Project Goal

The goal is to develop DocuAgent into a more robust AI document intelligence application, with persistent storage, improved retrieval quality, and agent-based workflows.

The project also serves as a hands-on demonstration of Python backend development, RAG architecture, API integration, and AI application testing.

---

**Development Status:** Active development – features and architecture are continuously being improved.
