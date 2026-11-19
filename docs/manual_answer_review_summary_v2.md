# Manuelle RAG-Antwortbewertung

Quellbericht: `answer_evaluation_live.json`
Reviewer: Manuelle Prüfung mit ChatGPT (GPT-6)

Generierte Antworten: **3**. Fälle mit mindestens einer Note: **3**.

| Kriterium | Mittelwert (0–2) | Bewertet | Unbewertet | Nicht anwendbar |
|---|---:|---:|---:|---:|
| Answer Correctness | 2.000 | 2 | 0 | 1 |
| Faithfulness | 2.000 | 3 | 0 | 0 |
| Citation Accuracy | 2.000 | 2 | 0 | 1 |
| Abstention | 2.000 | 1 | 0 | 2 |

Mittelwert = Summe der vergebenen Noten / Anzahl tatsächlich bewerteter Fälle pro Kriterium. Eine echte Note 0 zählt mit; unbewertet, ausgelassen und nicht anwendbar zählen nicht als 0. Ohne Noten bleibt der Mittelwert null bzw. nicht bewertet. Kein Gesamtwert über verschiedene Kriterien wird gebildet.

## Einzelbewertungen

| Frage | Kriterium | Status | Note | Begründung |
|---|---|---|---:|---|
| A01: Wie viele Urlaubstage haben Mitarbeiter pro Jahr? | Answer Correctness | scored | 2 | 30 bezahlte Urlaubstage bei einer Fünf-Tage-Woche und anteilige Berechnung bei Teilzeit stimmen mit der Sollantwort überein. |
| A01: Wie viele Urlaubstage haben Mitarbeiter pro Jahr? | Faithfulness | scored | 2 | Beide Aussagen sind im tatsächlich übergebenen Text von Chunk 2 auf Seite 1 enthalten. |
| A01: Wie viele Urlaubstage haben Mitarbeiter pro Jahr? | Citation Accuracy | scored | 2 | Die angegebene Quelle DocuAgent_Testdokument.pdf, Seite 1, Chunk 2 belegt die Antwort. |
| A01: Wie viele Urlaubstage haben Mitarbeiter pro Jahr? | Abstention | not_applicable | – | Gemäß Rubrik für diese Frage nicht anwendbar. |
| A11: Wie lang muss ein Passwort sein? | Answer Correctness | scored | 2 | Die Antwort nennt die korrekte Mindestlänge von 14 Zeichen. |
| A11: Wie lang muss ein Passwort sein? | Faithfulness | scored | 2 | Die Passwortregel ist im tatsächlich übergebenen Chunk 2 der IT-Sicherheitsrichtlinie enthalten. |
| A11: Wie lang muss ein Passwort sein? | Citation Accuracy | scored | 2 | Die angegebene Quelle DocuAgent_IT_Sicherheitsrichtlinie.pdf, Seite 1, Chunk 2 belegt die Antwort. |
| A11: Wie lang muss ein Passwort sein? | Abstention | not_applicable | – | Gemäß Rubrik für diese Frage nicht anwendbar. |
| A21: Wie hoch ist das monatliche Bruttogehalt der Mitarbeiter? | Answer Correctness | not_applicable | – | Gemäß Rubrik für diese Frage nicht anwendbar. |
| A21: Wie hoch ist das monatliche Bruttogehalt der Mitarbeiter? | Faithfulness | scored | 2 | Die Antwort nennt keinen unbelegten Gehaltswert und erklärt nur, dass die Information im vorliegenden Dokument nicht gefunden wurde. |
| A21: Wie hoch ist das monatliche Bruttogehalt der Mitarbeiter? | Citation Accuracy | not_applicable | – | Korrekte Verweigerung ohne Tatsachenbehauptung und ohne Quellen; gemäß Rubrik nicht anwendbar. |
| A21: Wie hoch ist das monatliche Bruttogehalt der Mitarbeiter? | Abstention | scored | 2 | Die beiden Test-PDFs enthalten laut Referenz keine Gehaltsangaben; das Modell verzichtet korrekt auf eine erfundene Zahl. |

## Zuordnung und Grenzen

Quellbericht SHA-256: `93d946af50cc02fc930fb7f159da54b065ed5814f04d824bbbda107a7df19b2b`
Rubrik SHA-256: `c3f5964a287a34f7fe37ad7ccd942ae3892696d1dc12a5a132b4f25ab045fcb6`

Die Noten stammen ausschließlich aus der separaten manuellen Review-Datei. Dieser Lauf erzeugt weder Antworten noch automatische LLM-Bewertungen. Geringe Bewertungsabdeckung und subjektive Urteile begrenzen die Aussagekraft. Es erfolgt kein Zugriff auf OpenAI, Retrieval, .env oder SQLite.
