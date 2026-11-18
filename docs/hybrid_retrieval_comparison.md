# Semantic Search und Hybrid Search – echter lokaler Vergleich

Modell: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`. Identischer Datensatz mit 20 Fragen, 2 PDFs und 18 Chunks.

| Methode | Hit Rate@1 | Hit Rate@3 | MRR |
|---|---:|---:|---:|
| Semantic | 75.00% | 95.00% | 0.862500 |
| Hybrid | 65.00% | 80.00% | 0.765833 |

| ID | Frage | Erwartete Quelle | Rang Semantic | Rang Hybrid | Änderung |
|---|---|---|---:|---:|---|
| Q01 | Wie viele Urlaubstage haben Mitarbeiter pro Jahr? | DocuAgent_Testdokument.pdf, S. 1: Urlaub und Arbeitszeit | 1 | 2 | schlechter |
| Q02 | Wie früh muss ich einen Urlaubsantrag einreichen? | DocuAgent_Testdokument.pdf, S. 1: Urlaubsanträge | 1 | 5 | schlechter |
| Q03 | Wie viele Stunden beträgt die reguläre Wochenarbeitszeit? | DocuAgent_Testdokument.pdf, S. 1: Arbeitszeit | 2 | 2 | gleich |
| Q04 | Wann müssen Lieferantenrechnungen bezahlt werden? | DocuAgent_Testdokument.pdf, S. 2: Rechnungen und Einkauf | 1 | 5 | schlechter |
| Q05 | Bei welchem Bestellwert ist die Freigabe der Einkaufsleitung erforderlich? | DocuAgent_Testdokument.pdf, S. 2: Freigabe von Bestellungen | 1 | 1 | gleich |
| Q06 | Ab welchem Betrag muss die Geschäftsführung einer Bestellung zusätzlich zustimmen? | DocuAgent_Testdokument.pdf, S. 2: Zustimmung der Geschäftsführung | 1 | 1 | gleich |
| Q07 | An welchem Wochentag wird der Projektstatusbericht erstellt? | DocuAgent_Testdokument.pdf, S. 2: Projektmanagement | 1 | 1 | gleich |
| Q08 | Welche Programmiersprache wird überwiegend für interne Automatisierungen verwendet? | DocuAgent_Testdokument.pdf, S. 3: Softwareentwicklung und Automatisierung | 1 | 1 | gleich |
| Q09 | Welche Prüfungen brauchen Änderungen am produktiven Code? | DocuAgent_Testdokument.pdf, S. 3: Code-Review und Tests | 1 | 1 | gleich |
| Q10 | Wer muss bei kritischen Entscheidungen über neue KI-Funktionen die Freigabe geben? | DocuAgent_Testdokument.pdf, S. 3: Prüfung neuer KI-Funktionen | 1 | 1 | gleich |
| Q11 | Wie lang muss ein Passwort sein? | DocuAgent_IT_Sicherheitsrichtlinie.pdf, S. 1: Passwörter und Anmeldung | 2 | 6 | schlechter |
| Q12 | Ist Mehrfaktor-Authentifizierung für alle Unternehmenskonten vorgeschrieben? | DocuAgent_IT_Sicherheitsrichtlinie.pdf, S. 1: Passwörter und Anmeldung; DocuAgent_Testdokument.pdf, S. 3: IT-Sicherheit und Datenverarbeitung | 1 | 1 | gleich |
| Q13 | Innerhalb welcher Frist muss ein verlorener Firmenlaptop gemeldet werden? | DocuAgent_IT_Sicherheitsrichtlinie.pdf, S. 1: Sicherheitsvorfälle melden; DocuAgent_Testdokument.pdf, S. 3: IT-Sicherheit und Datenverarbeitung | 4 | 4 | gleich |
| Q14 | Über welchen Kanal sollen Sicherheitsvorfälle gemeldet werden? | DocuAgent_IT_Sicherheitsrichtlinie.pdf, S. 1: Sicherheitsvorfälle melden | 2 | 2 | gleich |
| Q15 | Nach wie vielen Minuten ohne Benutzung müssen sich Dienstgeräte automatisch sperren? | DocuAgent_IT_Sicherheitsrichtlinie.pdf, S. 1: Geräte und Software | 1 | 1 | gleich |
| Q16 | Wie schnell müssen von der IT freigegebene Sicherheitsupdates installiert werden? | DocuAgent_IT_Sicherheitsrichtlinie.pdf, S. 1: Geräte und Software | 2 | 1 | besser |
| Q17 | Darf ich einen privaten USB-Stick an ein Dienstgerät anschließen? | DocuAgent_IT_Sicherheitsrichtlinie.pdf, S. 1: Geräte und Software | 1 | 1 | gleich |
| Q18 | Zu welcher Uhrzeit werden die zentralen Projektdaten automatisch gesichert? | DocuAgent_IT_Sicherheitsrichtlinie.pdf, S. 1: Datensicherung | 1 | 1 | gleich |
| Q19 | Wie lange werden die Datensicherungen aufbewahrt? | DocuAgent_IT_Sicherheitsrichtlinie.pdf, S. 1: Datensicherung | 1 | 1 | gleich |
| Q20 | Wie oft findet ein Wiederherstellungstest für die Datensicherung statt? | DocuAgent_IT_Sicherheitsrichtlinie.pdf, S. 1: Datensicherung | 1 | 1 | gleich |

Verbessert: 1; verschlechtert: 4; unverändert: 15.

Die reine Wortzerlegung gleicht deutsche Flexionen wie Urlaubsantrag/Urlaubsanträge bzw. Passwort/Passwörter nicht ab. Ohne Wortübereinstimmung erhält ein Chunk keinen BM25-Rangbeitrag. Die vollständigen Rangbeiträge stehen als semantic_rank und bm25_rank im JSON-Bericht. Die Parameter bleiben unabhängig vom Ergebnis unverändert; semantic bleibt der Anwendungsstandard.

## Vor der Evaluation festgelegtes Verfahren

BM25Okapi aus rank-bm25 mit k1=1,5, b=0,75, epsilon=0,25; Unicode-Wörter/Zahlen und casefold. Kein Stemming und keine Stoppwortliste. Nur Chunks mit mindestens einer Wortübereinstimmung werden in die BM25-Rangliste aufgenommen. Die vollständige semantische Rangliste wird mit dieser Liste gleichgewichtet kombiniert: RRF = Summe 1 / (60 + Rang). Ränge beginnen bei 1. Bei Gleichstand bleibt die semantische Reihenfolge erhalten. Keine Parameter wurden anhand dieser 20 Fragen angepasst.

`similarity_score` bleibt die unveränderte Kosinus-Ähnlichkeit. Hybrid sortiert nach dem separaten `rrf_score`. BM25-Score, RRF-Score und Kosinus-Ähnlichkeit sind verschiedene Größen und keine Wahrscheinlichkeiten. Alle Rohscores und Ranglisten stehen im JSON-Bericht.

Die Dokument-Embeddings werden einmal erzeugt und für beide Methoden wiederverwendet. Der BM25-Index wird aus denselben Chunks aufgebaut. Keine Änderungen an PDFs, Referenzen, SQLite-Nutzerdaten oder .env; keine LLM-Aufrufe.

Relevanz verlangt denselben Dateinamen, dieselbe Seite und alle Textbelege im Chunk; alternative belegte Quellen sind erlaubt. Beispielfragen ohne Antwort zählen nicht. MRR verwendet die vollständige Rangliste. Der kleine Testdatensatz erlaubt keine allgemeine Aussage, dass Hybrid immer besser ist. PDF-Beispielfragen können die Suche begünstigen. Die LLM-Antwortqualität wird nicht gemessen.

Quellen: [rank-bm25](https://github.com/dorianbrown/rank_bm25), [RRF-Originalarbeit](https://cormack.uwaterloo.ca/cormack/cormacksigir09-rrf.pdf).
