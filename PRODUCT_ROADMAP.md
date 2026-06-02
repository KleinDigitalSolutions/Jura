# Kanzlei Product Roadmap

## Phase 1: Einsatzbereit machen

- Lokale Python-3.12-Umgebung und Modal-CLI herstellen.
- Index lokal neu aufbauen und in `legal-rag-data` hochladen.
- Modal-Secrets setzen: `my-deepseek-secret`, optional `my-anthropic-secret`.
- Smoke-Tests nach Deployment: `/api/legal/stats`, `/api/legal/search`, `/api/legal/ask/enhanced`.
- UI nur mit Enhanced Retrieval betreiben, damit Query-Rewriting, Multi-Query-Suche und Citation-Checks greifen.

## Phase 2: Antwortqualität absichern

- Retrieval-Evaluation als echtes Gate einführen: Mindestwerte für Recall@5 und Rechtsgebiet-Accuracy.
- Mandantenfragen mit erwarteten Fundstellen als Kanzlei-Testset pflegen.
- Antworten zusätzlich gegen Quellen prüfen: jede Empfehlung muss an konkrete Fundstelle gebunden sein.
- Bei schwachem Kontext keine juristische Schlussfolgerung erzwingen; stattdessen Rückfragen und fehlende Unterlagen nennen.

## Phase 3: Kanzlei-Workflow

- Mandantenakte: Kontakt, Sachverhalt, Dokumente, Fristen, Gegner, Streitwert.
- Intake-Formulare je Rechtsgebiet, z. B. Arbeitsrecht, Mietrecht, Verkehr, Familienrecht.
- Anwalt-Review-Modus: KI-Antwort als Entwurf, nicht direkt als finale Beratung.
- Export: Gesprächsprotokoll, Quellenliste, offene Fragen und To-do-Liste als PDF.

## Phase 4: Betrieb & Compliance

- Authentifizierung, Rollen und Kanzlei-Mandanten-Trennung.
- Audit-Log für KI-Antworten, Quellen und Modellversionen.
- Datenschutzkonzept: Löschfristen, Verschlüsselung, AVV, keine unnötigen personenbezogenen Daten.
- Monitoring: Latenz, Fehler, Retrieval-Qualität, nicht beantwortbare Fragen.

## Phase 5: Professionalisierung

- Custom Domain, Branding und Kanzlei-spezifische Tonalität.
- Admin-Bereich für Rechtsgebiete, Prompts, Kanzlei-Wissen und Testfälle.
- Feedback-Schleife: Anwalt markiert Antworten als korrekt/unkorrekt; daraus entstehen neue Eval-Cases.
