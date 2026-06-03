# Technical Extensions

Notes on directions this project could take if developed further. These are architectural considerations, not a product roadmap.

## Quality Layer Depth

The current system has one detailed quality profile (ordinary employee termination). A production system would need profiles for every high-value legal workflow:

- **Employment law**: extraordinary termination, change termination, operational termination, notice periods, probation
- **Tenancy law**: rent defects and reduction, deposit, termination, utility billing
- **Contract law**: payment default, limitation periods, purchase defects, consumer protection
- **Corporate law**: GmbH director liability, insolvency filing duty, shareholder disputes
- **Criminal law**: fraud, embezzlement, bodily harm, statute of limitations
- **Administrative law**: objections, administrative court actions, deadlines

Each profile defines required norms, excluded false positives, allowed law families, and answer-focus rules. The architecture already supports this — it's a data expansion, not a code change.

## Retrieval Channel Separation

Currently, statutes and court decisions share one Qdrant collection and one retrieval path. A stronger architecture would:

1. Retrieve statutes and case law in **separate channels**
2. Validate each channel independently against the quality profile
3. Fuse results after validation, clearly marking each source as binding law, supporting case law, or contextual commentary

This would reduce cross-contamination (e.g., a high-scoring BGH decision dragging in unrelated norms through its text content).

## Citation Resolution

The current citation parser uses regex patterns for `§`, `Abs.`, and law abbreviations. German legal citation is complex — a robust resolver would need:

- Normalized handling of `§§ 280 Abs. 1, 3, 281 Abs. 1 BGB` as multiple distinct references
- Resolution of relative references ("Satz 2" within a paragraph context)
- Handling of law-name variations and abbreviation conflicts

This would improve both the answer audit (fewer false positives in citation checking) and the quality layer (more precise source matching).

## Answer Audit Enforcement

The answer audit currently returns structured results but doesn't block delivery. In a production setting:

- `fail` status should trigger a visible warning in the UI ("Prüfung erforderlich")
- High-severity issues (missing mandatory norms, phantom citations) should prevent the answer from being presented as a confident legal analysis
- The audit trail should be exportable alongside the answer

## Eval Set Scale

The current eval set has 15 cases (10 regression guards, 5 known gaps). Meaningful coverage for a real deployment would require 100–150 cases spanning all major legal areas, with human-reviewed source expectations and cross-domain contamination checks.
