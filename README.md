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

### Evaluation der RAG-Antwortqualität

`backend/scripts/evaluate_answers.py` verwendet eine eigene Speichersammlung mit den
beiden künstlichen PDFs, Semantic Search und `top_k=3`. Die SQLite-Nutzerdaten und
`.env` werden nicht geöffnet oder verändert. Die Antwortgenerierung erhält nur Frage
und gefundene Chunks, niemals Sollantworten oder Bewertungsmuster.

Der Datensatz [answer_questions.json](data/evaluation/answer_questions.json) enthält
20 überprüfbare Sollantworten mit Quellenverweisen auf die bestehende
Retrieval-Referenzdatei und vier manuell am vollständigen PDF-Text geprüfte
Nichtwissen-Fragen (Gehalt, IT-Telefonnummer, Passwortmanager, Umsatz). Alle
beantwortbaren Quellenbelege werden vor dem Lauf gegen die PDF-Extraktion validiert.

**Zwei getrennte Bewertungsebenen:**

| Kriterium | Deterministische Teilprüfung | Semantische Prüfung durch Mensch oder separat freigegebenen LLM-Judge |
| --- | --- | --- |
| Answer Correctness | Regex-Prüfung wichtiger Zahlen/Begriffe (`correctness_fact_match_proxy`) | Ist die gesamte Antwort richtig und vollständig, einschließlich Bedingungen? |
| Faithfulness | Prüfen, ob der Retrieval-Kontext den Sollbeleg enthält | Folgt jede Tatsachenbehauptung aus den tatsächlich übergebenen Chunks? |
| Citation Accuracy | IDs/Dateiname gegen Kontext prüfen; erwarteten Beleg in zitierten Chunks suchen | Belegt jede angegebene Quelle tatsächlich die zugehörige Aussage? |
| Abstention | Explizite Nichtwissen-Formulierung und leere Quellenliste erkennen | Wurde bei wirklich fehlender Information korrekt und ohne erfundene Zusätze verweigert? |

Die semantische [Review-Rubrik](data/evaluation/answer_review_rubric.json) definiert
pro Kriterium 0 (nicht erfüllt), 1 (teilweise) oder 2 (vollständig). Im Bericht
bleiben diese Werte `null`, bis sie separat bewertet werden. Das Skript ruft
**keinen LLM-Judge** auf. Ein Regex-Treffer kann auch in einer falschen Aussage
stehen; gültige Quellen-IDs beweisen keine Faithfulness. Bei einer Verweigerung
wegen eines Retrieval-Fehlers kann Faithfulness gut, Answer Correctness aber
schlecht sein. Bericht und Rubrik unterscheiden deshalb fehlenden Dokumentinhalt
von fehlendem Retrieval-Kontext.

**Zuerst Dry-Run aus `backend`:**

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_answers.py --mode dry-run
# Vorschau genau der späteren drei Live-Fragen:
.\.venv\Scripts\python.exe scripts/evaluate_answers.py --mode dry-run --ids A01 A11 A21
```

Standard ist `dry-run`. Er berechnet nur Retrieval und eine Kostenschätzung;
Antworten und Qualitätsraten bleiben unbewertet. Der ausgeführte Dry-Run umfasst
24 Fragen und findet für 19 von 20 beantwortbaren Fragen einen Sollbeleg im
Top-3-Kontext. **0 OpenAI-Aufrufe, 0 generierte Antworten.** Der
[JSON-Bericht](docs/answer_evaluation_dry-run.json) enthält Sollantworten, Quellen,
tatsächlichen Kontext und ausfüllbare Review-Felder. `--output` wählt einen anderen
Berichtspfad; vorhandene Berichte werden nicht überschrieben. Bei einem erneuten
Lauf muss ein neuer Ausgabepfad angegeben werden.

Mit `--workflow legacy` (Standard) wird die Antwort direkt durch den bisherigen
RAG-Service erzeugt. `--workflow langgraph` verwendet den vorhandenen Workflow
mit `retrieve`, `abstain` und `generate`. Beide Varianten erhalten exakt den
vorbereiteten Kontext derselben Evaluationssammlung: Semantic Search, Top-3,
identische Dokument-IDs, Chunks und Scores. Der Graph-Retriever gibt diese Treffer
zurück, statt eine zweite Suche oder neue Dokument-Embeddings zu berechnen.

Kostenlose Dry-Runs aus `backend`, mit jeweils neuem Berichtsnamen:

```powershell
$evaluationRun = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
.\.venv\Scripts\python.exe scripts/evaluate_answers.py --mode dry-run --workflow legacy --output "../docs/answer_eval_legacy_$evaluationRun.json"
.\.venv\Scripts\python.exe scripts/evaluate_answers.py --mode dry-run --workflow langgraph --output "../docs/answer_eval_langgraph_$evaluationRun.json"
```

Der Dry-Run führt in beiden Varianten ausschließlich die gemeinsame
Kontextvorbereitung aus, weder Graph noch Antwortgenerator; er misst keine
Antwortqualität. Erst ein ausdrücklich bestätigter Live-Lauf führt den gewählten
Antwortpfad aus. Bei leerer Trefferliste verzichtet LangGraph lokal ohne
Generatoraufruf; legacy nutzt dafür den bestehenden Service ohne OpenAI-Aufruf.
Die Aufrufzählung zählt dabei nur Generierungsversuche mit nichtleerem Kontext.
Das Falllimit und die Kostenschätzung bleiben konservativ auf höchstens drei
Fälle begrenzt. Beide Live-Varianten verlangen weiterhin einen Schlüssel in der
Prozessumgebung, auch bei leerem Kontext.

Dies ist ein Vergleich der Orchestrierung mit festem Retrieval-Kontext, kein
FastAPI-End-to-End-Test: produktiver SQLite-Index, HTTP-Fehlerabbildung und
`.env`-Laden sind nicht beteiligt. Metriken und Review-Auswertung bleiben gleich;
der Bericht enthält zusätzlich `workflow` und eine Beschreibung der Vergleichsgrenzen.
Der LangGraph-Standardbericht heißt `docs/answer_evaluation_langgraph_dry-run.json`
bzw. `docs/answer_evaluation_langgraph_live.json`; die Legacy-Dateinamen bleiben erhalten.

**Erst nach ausdrücklicher Freigabe: kontrollierter Live-Test mit höchstens drei Fragen:**

```powershell
# OPENAI_API_KEY muss bereits sicher in der Prozessumgebung gesetzt sein.
# Keine Schlüssel in Befehle, Quellcode oder Logs schreiben.
.\.venv\Scripts\python.exe scripts/evaluate_answers.py --mode live --confirm-live --max-calls 3 --ids A01 A11 A21
```

Das prüft Urlaub, Passwortlänge und eine nicht enthaltene Gehaltsinformation.
Ohne `--confirm-live` bricht Live vor der Modellvorbereitung ab. `--max-calls`
akzeptiert ausschließlich 1 bis 3. Ohne `--ids` werden höchstens die ersten drei
Fragen ausgeführt; weitere Fragen stehen als übersprungen im Bericht. Bei einem
API-Fehler endet die Generierung sofort, ohne automatische Wiederholung.
`generate_answer()` begrenzt jede Antwort auf 1.000 Outputtokens und deaktiviert
SDK-Wiederholungen. Der Live-Bericht heißt standardmäßig
`docs/answer_evaluation_live.json`. Bei fehlendem Umgebungs-Schlüssel wird keine
Anfrage gesendet; dieses Skript lädt keine `.env`.

Vor jedem Live-Lauf erscheinen Frage-IDs, Aufruflimit und Kostenschätzung. Die
Inputtokenzahl ist eine grobe UTF-8-Bytes/3-Heuristik einschließlich Prompt/Schema
plus Reserve; Output wird mit 1.000 Tokens pro Frage geschätzt. Verwendete Tarife
(geprüft 18.11.2026): 0,40 USD pro Million Inputtokens und 1,60 USD pro Million
Outputtokens für [gpt-4.1-mini](https://developers.openai.com/api/docs/models/gpt-4.1-mini).
Die Schätzung ist **keine Preisgarantie und keine harte Dollar-Ausgabengrenze**;
Tarife vor einem echten Lauf prüfen. Tatsächliche Kosten werden nicht erfunden
und bleiben im Bericht `null`, da der bestehende RAG-Service keine Usage-Daten
zurückgibt.

Für die Nachprüfung speichert der Bericht Fingerabdrücke von Datensätzen, Rubrik
und System-Prompt sowie den tatsächlich verwendeten Kontext. LLM-Antworten können
bei Wiederholung variieren. Teilmetriken haben explizite Nenner; Fehler und
ausgelassene Fragen werden separat gezählt. Manuelle Reviews sollen jede Bewertung
mit Textstellen begründen. Ein späterer LLM-Judge benötigt eine eigene Freigabe und
ein eigenes Kostenbudget. Ergebnisse auf diesen künstlichen PDFs müssen mit
unabhängigen Fragen bestätigt werden.

Validierung nach dieser Erweiterung: **231 Tests bestanden**. Alle Live-Pfade
wurden dabei ausschließlich mit gemockter OpenAI API geprüft.

### Manuelle Review-Auswertung ohne API-Aufrufe

`backend/scripts/review_answers.py` wertet ausschließlich lokale JSON-Dateien aus.
Es importiert weder RAG-Services noch OpenAI, lädt kein Modell und greift nicht auf
SQLite oder `.env` zu. Der ursprüngliche Antwortbericht bleibt unverändert.
Eine manuelle Bewertung steht in einer **separaten JSON-Datei** und wird mittels
SHA-256 an den exakten Quellbericht und die vorhandene
[0–2-Review-Rubrik](data/evaluation/answer_review_rubric.json) gebunden.
Nach einer Änderung des Quellberichts oder der Rubrik muss eine neue Review-Datei
erstellt werden; alte Bewertungen werden nicht stillschweigend neu zugeordnet.

Bereits vorbereitet: [manual_answer_reviews.json](docs/manual_answer_reviews.json)
für die drei vorhandenen generierten Antworten und eine
[unbewertete Zusammenfassung](docs/manual_answer_review_summary.md).
Es wurden **keine Noten automatisch vergeben**. Für einen anderen Bericht kann
aus `backend` eine neue Vorlage erzeugt werden:

```powershell
.\.venv\Scripts\python.exe scripts/review_answers.py init --report ../docs/answer_evaluation_live.json --reviews ../docs/meine_answer_reviews.json --reviewer "Mein Name"
```

In der neuen Datei `reviewer` angeben und pro Frage/Kriterium bearbeiten:

```json
"faithfulness": {
  "status": "scored",
  "score": 2,
  "reason": "Alle Aussagen werden durch die angegebenen Kontextstellen gestützt."
}
```

Diese Note ist nur ein Formatbeispiel, keine tatsächliche Bewertung einer Antwort.
Die Statuswerte bedeuten:

- `scored`: ganze Zahl **0, 1 oder 2** und eine konkrete Begründung erforderlich.
- `unreviewed`: noch nicht bewertet; `score` bleibt `null`.
- `not_applicable`: nicht anwendbar; `score` bleibt `null`, Begründung erforderlich.

Gemäß Rubrik ist Answer Correctness bei unbeantwortbaren Fragen nicht anwendbar,
Abstention bei beantwortbaren Fragen. Die Vorlage kennzeichnet diese Fälle bereits.
Citation Accuracy kann bei einer Verweigerung ohne Tatsachenbehauptungen als nicht
anwendbar markiert werden; das muss der Reviewer anhand der tatsächlichen Antwort
prüfen. Frage, Sollantwort, Quellen und Kontext sind im Quellbericht nachzulesen.
Dry-Run-Fälle ohne generierte Antwort sind nicht bewertbar und werden nicht in die
Vorlage übernommen. Unbekannte oder doppelte IDs, ungültige Noten und Kriteriennamen
werden abgelehnt. Ausgelassene Fragen/Kriterien gelten als unbewertet.

Auswertung in einen **neuen** Markdown-Bericht schreiben:

```powershell
.\.venv\Scripts\python.exe scripts/review_answers.py summarize --report ../docs/answer_evaluation_live.json --reviews ../docs/meine_answer_reviews.json --output ../docs/meine_answer_reviews_v1.md
```

Nach weiteren Bewertungen für `--output` einen neuen Namen wie `_v2.md` wählen.
Vorhandene Dateien werden grundsätzlich abgelehnt und niemals überschrieben.
Dies gilt auch, wenn versehentlich der Quellbericht oder die Review-Datei als
Ausgabepfad angegeben wird.

Der Bericht zeigt pro Kriterium Durchschnitt auf der Skala 0–2, Anzahl tatsächlich
bewerteter Fälle, unbewertete Fälle und nicht anwendbare Fälle. Einzelnoten und
Begründungen bleiben sichtbar. Nur `scored` fließt in den Mittelwert ein:
Noten 2 und 0 ergeben **1,0 bei zwei Bewertungen**, unabhängig davon, wie viele
andere Fälle unbewertet oder nicht anwendbar sind. Ohne Bewertungen bleibt der
Mittelwert `null` bzw. „nicht bewertet“. Es wird kein irreführender Gesamtwert
über alle Kriterien gebildet. Bewertungsabdeckung und subjektive Urteile müssen
bei der Interpretation berücksichtigt werden.

Die lokale Vorlage/Auswertung wurde ohne erneute Antwortgenerierung ausgeführt.
Es gab keine Live-Evaluation, keine LLM-Bewertung und keine Änderung vorhandener
JSON-Berichte. Die neuen Tests prüfen insbesondere Nenner, echte Nullnoten,
ungültige Noten, Quellenbericht-Zuordnung und Überschreibschutz.
Validierung: **252 Tests bestanden**; `pip check` meldet keine Probleme.

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
