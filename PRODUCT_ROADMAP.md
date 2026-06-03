# Kanzlei Product Roadmap

## Trust & Correctness Hardening Plan

Ziel: maximales Vertrauen für Kanzleien durch messbare Quellenbindung, reproduzierbare Evals und klare Haftungsgrenzen. Gesetzes-Scraping ist bewusst nicht Teil dieses Plans; neue Datenquellen werden später separat bewertet.

### Phase 1: Sofort - Vertrauen aufbauen

- [x] Source-Level Answer Auditor: Jede generierte Antwort wird nachträglich gegen die bereitgestellten Quellen geprüft.
- [ ] Audit-Ergebnis in der UI sichtbar machen: `pass`, `warn`, `fail`, Score, kritische Hinweise.
- [ ] Blocking-Regel: Bei `fail` keine anwaltlich klingende Sicherheit anzeigen, sondern "Prüfung erforderlich" und konkrete Audit-Gründe nennen.
- [ ] Kanzlei-Eval-Set mit den ersten 25 repräsentativen Fragen anlegen.

Akzeptanzkriterien:

- Jede Antwort enthält `answer_audit`.
- Materielle Aussagen ohne Quelle werden als `missing_claim_citation` markiert.
- Erwähnte Paragraphen, die nicht in den zitierten Quellen stehen, werden als `paragraph_not_in_cited_sources` markiert.
- Pflichtnormen aus `retrieval_plan.required_norms` müssen sichtbar in der Antwort behandelt werden.

### Phase 2: Kurzfristig - Qualität beweisbar machen

- Kanzlei-Eval-Set auf 100 Fragen ausbauen: `query`, `rechtsgebiet`, `risk_level`, `must_include`, `must_not_include`, `expected_structure`, `human_review_notes`.
- Eval Runner bauen: Enhanced Search gegen Eval-Set ausführen, JSON/Markdown-Report erzeugen, Regressionen blockieren.
- Quality Profiles erweitern: Mietmängel, Mietkündigung, Kaufmängel, Zahlungsverzug, GmbH-Geschäftsführerhaftung, Insolvenzantragspflicht, Betrugsvorwurf, Verwaltungswiderspruch, Verjährung.
- Separate Retrieval-Kanäle vorbereiten: Normen und Rechtsprechung getrennt abrufen, danach validiert fusionieren.

Akzeptanzkriterien:

- CI/Local Gate schlägt fehl, wenn `must_include` fehlt oder `must_not_include` erscheint.
- Profile enthalten Pflichtquellen, ausgeschlossene False Positives und Tests.
- Antworten unterscheiden Normtext, Rechtsprechung und anwaltliche Einordnung sichtbar.

### Phase 3: Mittelfristig - Workflow-Nutzen

- Memo-Export mit Antwort, Quellenanhang, offenen Fragen, Fristen und Anwalt-Handoff.
- Mandantenkontext strukturiert speichern: Sachverhalt, Parteien, Fristen, Dokumenthinweise.
- Source Appendix: Jede genutzte Quelle mit Paragraph, Titel, Stand, Kontexttyp und Auditstatus ausgeben.
- Health- und Observability-Endpunkte: Modell, Indexstand, Dokumentzahl, Audit-Fehlerquote, Latenzen.

### Phase 4: Compliance & Betrieb

- Audit Logs für Anfrage, Retrieval-Plan, Quellen, Modell, Antwort-Audit und Exportereignisse.
- Mandanten-/Tenant-Trennung vorbereiten, bevor echte Kanzleidaten dauerhaft gespeichert werden.
- Datenretention und Löschpfade definieren.
- Kosten- und Provider-Fallback-Tracking ergänzen.

### Techstack-Entscheidungen

- Qdrant bleibt passend für Hybrid Retrieval: aktuelle Qdrant-Dokumentation empfiehlt die Kombination aus dense und sparse Repräsentationen sowie RRF/Query API für Hybrid Search.
- Gemini bleibt geeignet für strukturierte Audit-/Extraktionsaufgaben, weil die aktuelle Gemini API strukturierte Ausgaben per JSON Schema/Pydantic unterstützt.
- RAGAS ist als spätere Eval-Ergänzung sinnvoll für Faithfulness, Context Precision/Recall und Response Relevancy; das erste Gate bleibt aber deterministic und ohne zusätzliche LLM-Kosten.
- Modal ASGI und StreamingResponse bleiben passend für die bestehende FastAPI-App und SSE-Antworten.

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
