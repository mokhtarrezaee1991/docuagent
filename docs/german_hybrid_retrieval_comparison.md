# Retrieval-Vergleich mit deutscher Snowball-Normalisierung

Modell: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`. Identischer Datensatz mit 20 Fragen, 2 PDFs und 18 Chunks.

| Methode | Hit Rate@1 | Hit Rate@3 | MRR |
|---|---:|---:|---:|
| Semantic | 75.00% | 95.00% | 0.862500 |
| Hybrid bisher | 65.00% | 80.00% | 0.765833 |
| Hybrid DE | 60.00% | 95.00% | 0.776667 |

| ID | Frage | Erwartete Quelle | Semantic | Hybrid bisher | Hybrid DE | DE vs. Semantic | DE vs. Hybrid |
|---|---|---|---:|---:|---:|---|---|
| Q01 | Wie viele Urlaubstage haben Mitarbeiter pro Jahr? | DocuAgent_Testdokument.pdf, S. 1: Urlaub und Arbeitszeit | 1 | 2 | 2 | schlechter | gleich |
| Q02 | Wie früh muss ich einen Urlaubsantrag einreichen? | DocuAgent_Testdokument.pdf, S. 1: Urlaubsanträge | 1 | 5 | 2 | schlechter | besser |
| Q03 | Wie viele Stunden beträgt die reguläre Wochenarbeitszeit? | DocuAgent_Testdokument.pdf, S. 1: Arbeitszeit | 2 | 2 | 2 | gleich | gleich |
| Q04 | Wann müssen Lieferantenrechnungen bezahlt werden? | DocuAgent_Testdokument.pdf, S. 2: Rechnungen und Einkauf | 1 | 5 | 5 | schlechter | gleich |
| Q05 | Bei welchem Bestellwert ist die Freigabe der Einkaufsleitung erforderlich? | DocuAgent_Testdokument.pdf, S. 2: Freigabe von Bestellungen | 1 | 1 | 2 | schlechter | schlechter |
| Q06 | Ab welchem Betrag muss die Geschäftsführung einer Bestellung zusätzlich zustimmen? | DocuAgent_Testdokument.pdf, S. 2: Zustimmung der Geschäftsführung | 1 | 1 | 1 | gleich | gleich |
| Q07 | An welchem Wochentag wird der Projektstatusbericht erstellt? | DocuAgent_Testdokument.pdf, S. 2: Projektmanagement | 1 | 1 | 1 | gleich | gleich |
| Q08 | Welche Programmiersprache wird überwiegend für interne Automatisierungen verwendet? | DocuAgent_Testdokument.pdf, S. 3: Softwareentwicklung und Automatisierung | 1 | 1 | 1 | gleich | gleich |
| Q09 | Welche Prüfungen brauchen Änderungen am produktiven Code? | DocuAgent_Testdokument.pdf, S. 3: Code-Review und Tests | 1 | 1 | 1 | gleich | gleich |
| Q10 | Wer muss bei kritischen Entscheidungen über neue KI-Funktionen die Freigabe geben? | DocuAgent_Testdokument.pdf, S. 3: Prüfung neuer KI-Funktionen | 1 | 1 | 1 | gleich | gleich |
| Q11 | Wie lang muss ein Passwort sein? | DocuAgent_IT_Sicherheitsrichtlinie.pdf, S. 1: Passwörter und Anmeldung | 2 | 6 | 2 | gleich | besser |
| Q12 | Ist Mehrfaktor-Authentifizierung für alle Unternehmenskonten vorgeschrieben? | DocuAgent_IT_Sicherheitsrichtlinie.pdf, S. 1: Passwörter und Anmeldung; DocuAgent_Testdokument.pdf, S. 3: IT-Sicherheit und Datenverarbeitung | 1 | 1 | 1 | gleich | gleich |
| Q13 | Innerhalb welcher Frist muss ein verlorener Firmenlaptop gemeldet werden? | DocuAgent_IT_Sicherheitsrichtlinie.pdf, S. 1: Sicherheitsvorfälle melden; DocuAgent_Testdokument.pdf, S. 3: IT-Sicherheit und Datenverarbeitung | 4 | 4 | 3 | besser | besser |
| Q14 | Über welchen Kanal sollen Sicherheitsvorfälle gemeldet werden? | DocuAgent_IT_Sicherheitsrichtlinie.pdf, S. 1: Sicherheitsvorfälle melden | 2 | 2 | 2 | gleich | gleich |
| Q15 | Nach wie vielen Minuten ohne Benutzung müssen sich Dienstgeräte automatisch sperren? | DocuAgent_IT_Sicherheitsrichtlinie.pdf, S. 1: Geräte und Software | 1 | 1 | 1 | gleich | gleich |
| Q16 | Wie schnell müssen von der IT freigegebene Sicherheitsupdates installiert werden? | DocuAgent_IT_Sicherheitsrichtlinie.pdf, S. 1: Geräte und Software | 2 | 1 | 1 | besser | gleich |
| Q17 | Darf ich einen privaten USB-Stick an ein Dienstgerät anschließen? | DocuAgent_IT_Sicherheitsrichtlinie.pdf, S. 1: Geräte und Software | 1 | 1 | 1 | gleich | gleich |
| Q18 | Zu welcher Uhrzeit werden die zentralen Projektdaten automatisch gesichert? | DocuAgent_IT_Sicherheitsrichtlinie.pdf, S. 1: Datensicherung | 1 | 1 | 1 | gleich | gleich |
| Q19 | Wie lange werden die Datensicherungen aufbewahrt? | DocuAgent_IT_Sicherheitsrichtlinie.pdf, S. 1: Datensicherung | 1 | 1 | 1 | gleich | gleich |
| Q20 | Wie oft findet ein Wiederherstellungstest für die Datensicherung statt? | DocuAgent_IT_Sicherheitsrichtlinie.pdf, S. 1: Datensicherung | 1 | 1 | 1 | gleich | gleich |

Hybrid DE gegenüber semantic: verbessert 2, verschlechtert 4, unverändert 14.
Hybrid DE gegenüber hybrid: verbessert 3, verschlechtert 1, unverändert 16.

## Wortverarbeitung und Grenzen

Snowball-Paketversion: 3.1.1. Dokumente und Fragen verwenden dieselbe Normalisierung.

| Wörter | Bisherige Tokens | Deutsche Snowball-Stämme |
|---|---|---|
| Passwort / Passwörter | passwort / passwörter | passwort / passwort |
| Urlaubsantrag / Urlaubsanträge | urlaubsantrag / urlaubsanträge | urlaubsantrag / urlaubsantrag |
| einreichen / eingereicht | einreichen / eingereicht | einreich / eingereicht |
| Mitarbeiter / Mitarbeitenden | mitarbeiter / mitarbeitenden | mitarbeit / mitarbeit |
| Rechnung / Lieferantenrechnungen | rechnung / lieferantenrechnungen | rechnung / lieferantenrechn |
| Urlaub / Urlaubstage | urlaub / urlaubstage | urlaub / urlaubstag |
| muss / müssen | muss / müssen | muss / muss |

Snowball gleicht einige Flexionsformen und Umlaute an, ist aber kein vollständiger Lemmatizer. Zusammengesetzte Wörter werden nicht zerlegt: Rechnung und Lieferantenrechnungen sowie Urlaub und Urlaubstage bleiben unterschiedliche Tokens. Auch unregelmäßige Verbformen müssen nicht zusammenfallen. Neue Stämme verändern Wortfrequenzen und IDF-Werte im gesamten BM25-Korpus, deshalb sind auch Verschlechterungen möglich. Es gibt keine manuell auf Testfragen zugeschnittenen Wortlisten oder Synonymregeln.

## Analyse der vier bisherigen Verschlechterungen

BM25-Ränge beziehen sich auf den jeweils ersten relevanten Chunk; – bedeutet keinen Worttreffer.

| ID | BM25-Rang bisher | Trefferwörter bisher | BM25-Rang DE | Trefferstämme DE |
|---|---:|---|---:|---|
| Q01 | 3 | urlaubstage, pro | 3 | urlaubstag, pro |
| Q02 | – | – | 3 | muss, ein, urlaubsantrag |
| Q04 | 13 | werden | 12 | muss, werd |
| Q11 | – | – | 3 | muss, passwort |

## Veränderte Ränge gegenüber bisherigem Hybrid

| ID | Rang bisher → DE | BM25-Rang des relevanten Treffers bisher → DE |
|---|---|---|
| Q02 | 5 → 2 | – → 3 |
| Q05 | 1 → 2 | 2 → 3 |
| Q11 | 6 → 2 | – → 3 |
| Q13 | 4 → 3 | 5 → 4 |

## Vor der Evaluation festgelegtes Verfahren

BM25Okapi aus rank-bm25 mit k1=1,5, b=0,75, epsilon=0,25; Unicode-Wörter/Zahlen und casefold. Bisheriger Hybrid-Modus unverändert ohne Stemming; Hybrid DE zusätzlich mit Unicode-NFC und deutschem Snowball-Stemming. Keine Stoppwortliste und keine Kompositazerlegung in beiden Varianten. Nur Chunks mit mindestens einer Wortübereinstimmung werden in die BM25-Rangliste aufgenommen. Die vollständige semantische Rangliste wird mit dieser Liste gleichgewichtet kombiniert: RRF = Summe 1 / (60 + Rang). Ränge beginnen bei 1. Bei Gleichstand bleibt die semantische Reihenfolge erhalten. Keine Parameter wurden anhand dieser 20 Fragen angepasst.

`similarity_score` bleibt die unveränderte Kosinus-Ähnlichkeit. Hybrid sortiert nach dem separaten `rrf_score`. BM25-Score, RRF-Score und Kosinus-Ähnlichkeit sind verschiedene Größen und keine Wahrscheinlichkeiten. Alle Rohscores und Ranglisten stehen im JSON-Bericht.

Die Dokument-Embeddings werden einmal erzeugt und für beide Methoden wiederverwendet. Der BM25-Index wird aus denselben Chunks aufgebaut. Keine Änderungen an PDFs, Referenzen, SQLite-Nutzerdaten oder .env; keine LLM-Aufrufe.

Relevanz verlangt denselben Dateinamen, dieselbe Seite und alle Textbelege im Chunk; alternative belegte Quellen sind erlaubt. Beispielfragen ohne Antwort zählen nicht. MRR verwendet die vollständige Rangliste. Der kleine Testdatensatz erlaubt keine allgemeine Aussage, dass Hybrid immer besser ist. PDF-Beispielfragen können die Suche begünstigen. Verbesserungen müssen mit unabhängigen Fragen bestätigt werden. Semantic bleibt der Standard. Die LLM-Antwortqualität wird nicht gemessen.

Quellen: [rank-bm25](https://github.com/dorianbrown/rank_bm25), [RRF-Originalarbeit](https://cormack.uwaterloo.ca/cormack/cormacksigir09-rrf.pdf).

Deutsche Normalisierung: [Snowball-Algorithmus](https://snowballstem.org/algorithms/german/stemmer.html).
