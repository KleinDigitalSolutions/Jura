"""Post-generation source-level answer auditing.

The retrieval quality layer decides which sources may reach the LLM. This
module checks the generated answer against those sources afterwards. It is
intentionally deterministic so it can run in production and tests without an
extra judge model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


LAW_PATTERN = re.compile(
    r"\b(BGB|StGB|HGB|ZPO|StPO|GG|VwVfG|AktG|GmbHG?|InsO|FamFG|"
    r"BDSG|UrhG|MarkenG|PatG|BauGB|VwGO|AO|KStG|EStG|UStG|UmwG|WpHG|BetrVG|"
    r"SGB\s*IX|SGB|KSchG|BVerfGG|TKG|WEG|EGBGB|BGBEG|UWG|TTDSG|TDDDG)\b",
    re.IGNORECASE,
)
PARA_PATTERN = re.compile(r"§+\s*(\d+[a-z]?)", re.IGNORECASE)
CITATION_PATTERN = re.compile(r"\[(\d+)\]")
CITATION_BLOCK_PATTERN = re.compile(r"\[([0-9,\s]+)\]")

MATERIAL_TERMS = (
    "anspruch",
    "arbeitgeber",
    "arbeitnehmer",
    "betriebsrat",
    "beweislast",
    "darf",
    "frist",
    "haftung",
    "kündigung",
    "mangel",
    "muss",
    "pflicht",
    "recht",
    "rechtsfolge",
    "schadensersatz",
    "schriftform",
    "unwirksam",
    "verjähr",
    "vertrag",
    "wirksam",
    "zulässig",
)

OVERCONFIDENT_PATTERNS = (
    re.compile(r"\bimmer\b", re.IGNORECASE),
    re.compile(r"\bgarantiert\b", re.IGNORECASE),
    re.compile(r"\bohne\s+jede[nr]?\s+zweifel\b", re.IGNORECASE),
    re.compile(r"\bzweifelsfrei\b", re.IGNORECASE),
    re.compile(r"\bin\s+jedem\s+fall\b", re.IGNORECASE),
)


@dataclass
class ClaimIssue:
    """A single answer audit issue."""

    issue: str
    severity: str
    claim: str
    detail: str
    citation_ids: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "issue": self.issue,
            "severity": self.severity,
            "claim": self.claim,
            "detail": self.detail,
            "citation_ids": self.citation_ids,
        }


def _normalize_law(value: str) -> str:
    text = (value or "").upper().replace(" ", "")
    if text == "GMBH":
        return "GMBHG"
    return text


def _normalize_para(value: str) -> str:
    match = PARA_PATTERN.search(value or "")
    return match.group(1).lower() if match else ""


def _split_claims(answer: str) -> list[str]:
    """Split answer into sentence-like claims while keeping markdown bullets."""
    normalized = re.sub(r"\r\n?", "\n", answer or "")
    normalized = re.sub(r"(\d+)\.", r"\1__DOT__", normalized)
    abbreviations = {
        "z.B.": "z__DOT__B__DOT__",
        "d.h.": "d__DOT__h__DOT__",
        "bzw.": "bzw__DOT__",
        "ggf.": "ggf__DOT__",
        "ca.": "ca__DOT__",
    }
    for original, protected in abbreviations.items():
        normalized = normalized.replace(original, protected)
    raw_parts = re.split(r"(?<=[.!?])\s+|\n+", normalized)
    claims: list[str] = []
    skip_summary = False
    for part in raw_parts:
        claim = part.strip()
        claim = claim.replace("__DOT__", ".")
        for original, protected in abbreviations.items():
            claim = claim.replace(protected, original)
        claim = re.sub(r"^[-*]\s+", "", claim)
        claim = re.sub(r"^#{1,6}\s*", "", claim)
        if not claim or claim in {"---"}:
            continue
        plain = re.sub(r"[*_`]", "", claim).strip().lower()
        if "zusammenfassende kernaussage" in plain or plain.startswith("zusammenfassend"):
            skip_summary = True
            continue
        if plain.startswith("dies ist eine allgemeine information"):
            skip_summary = False
        elif skip_summary:
            continue
        claims.append(claim)
    return claims


def _extract_citation_ids(claim: str) -> list[str]:
    """Extract citation ids from [1], [1, 3], and [1,3] style references."""
    ids: list[str] = []
    for block in CITATION_BLOCK_PATTERN.findall(claim or ""):
        for raw_id in re.split(r"\s*,\s*", block.strip()):
            if raw_id and raw_id.isdigit() and raw_id not in ids:
                ids.append(raw_id)
    return ids


def _is_material_claim(claim: str) -> bool:
    text = claim.strip().lower()
    plain = re.sub(r"[*_`]", "", text).strip()
    if len(text) < 18:
        return False
    if re.match(r"^(?:\d+\.\s*)?(issue|rule|analysis|conclusion)\b", plain):
        return False
    if re.match(r"^(?:\d+\.\s*)?(prüfung|handlungsempfehlung|zusammenfassend)\b", plain):
        return False
    if text.startswith(("sehr geehrte", "gerne erläutere", "auf basis der")):
        return False
    if text.startswith(("hinweis zur quellenprüfung", "bitte überprüfen")):
        return False
    if text.startswith(("dies ist eine allgemeine information", "bitte stellen", "bitte reichen", "bitte bringen")):
        return False
    if text.startswith(("für eine abschließende beurteilung", "um die wirksamkeit", "für die wirksamkeit")):
        return False
    if text.startswith(("es ist ratsam", "für bestimmte arbeitnehmergruppen kann besonderer kündigungsschutz bestehen")):
        return False
    if text.startswith("ein anwalt"):
        return False
    if re.match(r"^die wirksamkeit .* richtet sich nach verschiedenen gesetzlichen", plain):
        return False
    if not _extract_citation_ids(claim) and re.match(r"^\*\*[^*]+:\*\*", claim.strip()):
        return False
    if not _extract_citation_ids(claim) and re.match(r"^\d+\.\s+\*\*[^*]+:\*\*", claim.strip()):
        return False
    if plain.endswith(":"):
        return False
    if text.endswith("?"):
        return False
    if _extract_citation_ids(claim) or PARA_PATTERN.search(claim) or LAW_PATTERN.search(claim):
        return True
    return any(term in text for term in MATERIAL_TERMS)


def _citation_lookup(citations: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for idx, citation in enumerate(citations, 1):
        raw_id = str(citation.get("id") or f"[{idx}]")
        match = CITATION_PATTERN.search(raw_id)
        cid = match.group(1) if match else str(idx)
        lookup[cid] = citation
    return lookup


def _citation_matches_norm(citation: dict[str, Any], law: str, para: str) -> bool:
    source_law = _normalize_law(str(citation.get("gesetz", "") or citation.get("abkürzung", "")))
    source_para = _normalize_para(str(citation.get("paragraph", "")))
    return source_law == law and source_para == para


def _required_norm_is_answered(
    answer: str,
    label: str,
    citation_by_id: dict[str, dict[str, Any]] | None = None,
) -> bool:
    law_match = LAW_PATTERN.search(label or "")
    para = _normalize_para(label)
    if not law_match or not para:
        return False
    law = _normalize_law(law_match.group(1))
    answer_laws = {_normalize_law(m.group(1)) for m in LAW_PATTERN.finditer(answer or "")}
    answer_paras = {_normalize_para(m.group(0)) for m in PARA_PATTERN.finditer(answer or "")}
    if law in answer_laws and para in answer_paras:
        return True

    citations = citation_by_id or {}
    return any(
        _citation_matches_norm(citations[cid], law, para)
        for cid in _extract_citation_ids(answer)
        if cid in citations
    )


def _profile_deadline_issue(answer: str, plan_data: dict[str, Any]) -> str:
    profiles = plan_data.get("profiles") or []
    required = set(plan_data.get("required_norms") or [])
    answer_l = (answer or "").lower()
    if (
        "arbeitsrecht_ordentliche_kuendigung_arbeitnehmer" in profiles
        and "KSchG § 4" in required
        and not re.search(r"(drei|3)[-\s]?(wochen|wöch)|kündigungsschutzklagefrist|§\s*4", answer_l)
    ):
        return "KSchG § 4 verlangt die Kündigungsschutzklagefrist; die Antwort nennt diese Frist nicht klar."
    return ""


def audit_answer_sources(
    answer: str,
    citations: list[dict[str, Any]],
    retrieval_plan: dict[str, Any] | None = None,
    source_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Audit generated answer against the citations sent to the LLM.

    The audit is conservative: it does not prove that every legal claim is
    correct, but it flags answer text that lacks a cited source, cites a source
    that was not provided, mentions paragraphs not present in the cited source,
    omits required norms, or uses risky overconfident language.
    """
    citation_by_id = _citation_lookup(citations)
    provided_ids = set(citation_by_id)
    issues: list[ClaimIssue] = []
    material_claims = 0
    cited_claims = 0

    for claim in _split_claims(answer):
        if not _is_material_claim(claim):
            continue
        material_claims += 1
        claim_citation_ids = _extract_citation_ids(claim)
        if claim_citation_ids:
            cited_claims += 1
        else:
            issues.append(
                ClaimIssue(
                    issue="missing_claim_citation",
                    severity="high",
                    claim=claim,
                    detail="Materielle rechtliche Aussage ohne Quellenverweis.",
                )
            )
            continue

        invalid_ids = [cid for cid in claim_citation_ids if cid not in provided_ids]
        if invalid_ids:
            issues.append(
                ClaimIssue(
                    issue="invalid_citation",
                    severity="high",
                    claim=claim,
                    detail="Quellenverweis wurde dem Modell nicht als Kontext bereitgestellt.",
                    citation_ids=[f"[{cid}]" for cid in invalid_ids],
                )
            )

        cited_sources = [citation_by_id[cid] for cid in claim_citation_ids if cid in citation_by_id]
        cited_paras: set[str] = set()
        for source in cited_sources:
            para = _normalize_para(str(source.get("paragraph", "")))
            if para:
                cited_paras.add(para)
            source_text = " ".join(
                str(source.get(key, ""))
                for key in ("titel", "text_preview")
            )
            cited_paras.update(
                _normalize_para(m.group(0))
                for m in PARA_PATTERN.finditer(source_text)
                if _normalize_para(m.group(0))
            )
        cited_laws = {
            _normalize_law(str(source.get("gesetz", "") or source.get("abkürzung", "")))
            for source in cited_sources
            if source.get("gesetz") or source.get("abkürzung")
        }

        claim_paras = {_normalize_para(m.group(0)) for m in PARA_PATTERN.finditer(claim)}
        ungrounded_paras = sorted(p for p in claim_paras if p and p not in cited_paras)
        if ungrounded_paras:
            issues.append(
                ClaimIssue(
                    issue="paragraph_not_in_cited_sources",
                    severity="high",
                    claim=claim,
                    detail="Genannte Paragraphen stehen nicht in den Quellen, die derselbe Satz zitiert.",
                    citation_ids=[f"[{cid}]" for cid in claim_citation_ids],
                )
            )

        claim_laws = {_normalize_law(m.group(1)) for m in LAW_PATTERN.finditer(claim)}
        ungrounded_laws = sorted(law for law in claim_laws if law and cited_laws and law not in cited_laws)
        if claim_paras and ungrounded_laws:
            issues.append(
                ClaimIssue(
                    issue="law_not_in_cited_sources",
                    severity="medium",
                    claim=claim,
                    detail="Genannte Gesetzesabkürzung passt nicht zu den Quellen, die derselbe Satz zitiert.",
                    citation_ids=[f"[{cid}]" for cid in claim_citation_ids],
                )
            )

        for pattern in OVERCONFIDENT_PATTERNS:
            if pattern.search(claim):
                issues.append(
                    ClaimIssue(
                        issue="overconfident_language",
                        severity="medium",
                        claim=claim,
                        detail="Kanzlei-taugliche Antworten sollten absolute Aussagen vermeiden, sofern der Kontext keine Vollprüfung trägt.",
                        citation_ids=[f"[{cid}]" for cid in claim_citation_ids],
                    )
                )
                break

    plan_data = retrieval_plan or {}
    source_audit_data = source_audit or {}
    missing_in_retrieval = set(source_audit_data.get("missing_required") or [])
    missing_required_in_answer = [
        label
        for label in plan_data.get("required_norms") or []
        if label not in missing_in_retrieval and not _required_norm_is_answered(answer, label, citation_by_id)
    ]
    for label in missing_required_in_answer:
        issues.append(
            ClaimIssue(
                issue="missing_required_norm_in_answer",
                severity="high",
                claim=label,
                detail="Pflichtnorm aus dem Retrieval-Plan wurde in der Antwort nicht sichtbar behandelt.",
            )
        )

    deadline_detail = _profile_deadline_issue(answer, plan_data)
    if deadline_detail:
        issues.append(
            ClaimIssue(
                issue="missing_deadline",
                severity="high",
                claim="Kündigungsschutzklagefrist",
                detail=deadline_detail,
            )
        )

    high_count = sum(1 for issue in issues if issue.severity == "high")
    medium_count = sum(1 for issue in issues if issue.severity == "medium")
    score = max(0, 100 - high_count * 20 - medium_count * 8)
    status = "pass"
    if high_count:
        status = "fail"
    elif medium_count:
        status = "warn"

    return {
        "status": status,
        "score": score,
        "material_claims": material_claims,
        "cited_claims": cited_claims,
        "issue_count": len(issues),
        "high_severity_count": high_count,
        "medium_severity_count": medium_count,
        "issues": [issue.as_dict() for issue in issues],
        "missing_required_in_answer": missing_required_in_answer,
        "retrieval_missing_required": sorted(missing_in_retrieval),
    }
