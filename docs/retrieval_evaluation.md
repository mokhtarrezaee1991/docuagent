# Retrieval-Evaluation – tatsächlicher lokaler Modelllauf

Modell: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`; Sentence Transformers: `6.3.0`.
Datensatz: 20 Fragen; 2 künstliche PDFs; 18 Chunks.

- Hit Rate@1: **75.00%**
- Hit Rate@3: **95.00%**
- MRR (vollständige Rangliste): **0.862500**

Dateikürzel: H = DocuAgent_Testdokument.pdf; IT = DocuAgent_IT_Sicherheitsrichtlinie.pdf.

| ID | Frage | Erwartete Quelle | Rang | Score des ersten relevanten Chunks |
|---|---|---|---:|---:|
| Q01 | Wie viele Urlaubstage haben Mitarbeiter pro Jahr? | H, S. 1: Urlaub und Arbeitszeit | 1 | 0.710757 |
| Q02 | Wie früh muss ich einen Urlaubsantrag einreichen? | H, S. 1: Urlaubsanträge | 1 | 0.608669 |
| Q03 | Wie viele Stunden beträgt die reguläre Wochenarbeitszeit? | H, S. 1: Arbeitszeit | 2 | 0.671629 |
| Q04 | Wann müssen Lieferantenrechnungen bezahlt werden? | H, S. 2: Rechnungen und Einkauf | 1 | 0.843195 |
| Q05 | Bei welchem Bestellwert ist die Freigabe der Einkaufsleitung erforderlich? | H, S. 2: Freigabe von Bestellungen | 1 | 0.657289 |
| Q06 | Ab welchem Betrag muss die Geschäftsführung einer Bestellung zusätzlich zustimmen? | H, S. 2: Zustimmung der Geschäftsführung | 1 | 0.583128 |
| Q07 | An welchem Wochentag wird der Projektstatusbericht erstellt? | H, S. 2: Projektmanagement | 1 | 0.739999 |
| Q08 | Welche Programmiersprache wird überwiegend für interne Automatisierungen verwendet? | H, S. 3: Softwareentwicklung und Automatisierung | 1 | 0.653698 |
| Q09 | Welche Prüfungen brauchen Änderungen am produktiven Code? | H, S. 3: Code-Review und Tests | 1 | 0.620760 |
| Q10 | Wer muss bei kritischen Entscheidungen über neue KI-Funktionen die Freigabe geben? | H, S. 3: Prüfung neuer KI-Funktionen | 1 | 0.606635 |
| Q11 | Wie lang muss ein Passwort sein? | IT, S. 1: Passwörter und Anmeldung | 2 | 0.510444 |
| Q12 | Ist Mehrfaktor-Authentifizierung für alle Unternehmenskonten vorgeschrieben? | IT, S. 1: Passwörter und Anmeldung; H, S. 3: IT-Sicherheit und Datenverarbeitung | 1 | 0.523674 |
| Q13 | Innerhalb welcher Frist muss ein verlorener Firmenlaptop gemeldet werden? | IT, S. 1: Sicherheitsvorfälle melden; H, S. 3: IT-Sicherheit und Datenverarbeitung | 4 | 0.480269 |
| Q14 | Über welchen Kanal sollen Sicherheitsvorfälle gemeldet werden? | IT, S. 1: Sicherheitsvorfälle melden | 2 | 0.567248 |
| Q15 | Nach wie vielen Minuten ohne Benutzung müssen sich Dienstgeräte automatisch sperren? | IT, S. 1: Geräte und Software | 1 | 0.788270 |
| Q16 | Wie schnell müssen von der IT freigegebene Sicherheitsupdates installiert werden? | IT, S. 1: Geräte und Software | 2 | 0.667465 |
| Q17 | Darf ich einen privaten USB-Stick an ein Dienstgerät anschließen? | IT, S. 1: Geräte und Software | 1 | 0.521561 |
| Q18 | Zu welcher Uhrzeit werden die zentralen Projektdaten automatisch gesichert? | IT, S. 1: Datensicherung | 1 | 0.701382 |
| Q19 | Wie lange werden die Datensicherungen aufbewahrt? | IT, S. 1: Datensicherung | 1 | 0.713140 |
| Q20 | Wie oft findet ein Wiederherstellungstest für die Datensicherung statt? | IT, S. 1: Datensicherung | 1 | 0.605047 |

## Messverfahren und Grenzen

Ein Treffer zählt nur, wenn Dateiname, Seite und sämtliche angegebenen Belegstellen im selben Chunk vorkommen. Reine Beispielfragen zählen nicht. Alternative gültige Quellen werden ausdrücklich berücksichtigt. Alle Referenzen werden vorab gegen den extrahierten PDF-Text geprüft; Zeichenumbrüche und Großschreibung werden normalisiert.

Hit Rate@1/3 misst den Anteil der Fragen mit mindestens einem relevanten Treffer unter den ersten 1/3. MRR mittelt 1/Rang des ersten relevanten Treffers; fehlende Treffer zählen 0. Es wird über alle Chunks gesucht, nicht nur bis Rang 3. Scores und Rangfolgen werden nicht verändert.

Der kleine, manuell erstellte Datensatz ist kein allgemeiner Qualitätsnachweis. Einzelne PDF-Beispielfragen ähneln Evaluationsfragen und können die Suche begünstigen. Strikte Textbelege können auf mehrere Chunks verteilte Antworten als Fehlschlag werten. Gemessen wird nur Retrieval, nicht die Richtigkeit einer generierten Antwort.

Der JSON-Bericht enthält Belegstellen, vollständige Rangfolgen, Dokument-IDs und SHA-256-Fingerabdrücke von PDFs und Datensatz. Er wird bei erneutem Ausführen überschrieben. Die Evaluation verwendet eine eigene In-Memory-Sammlung und verändert keine SQLite-Nutzerdaten. Es erfolgen keine OpenAI-Aufrufe und kein Zugriff auf .env.
