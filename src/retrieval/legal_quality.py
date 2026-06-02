"""Deterministic legal quality layer for retrieval and answer generation.

This module keeps legal retrieval requirements in code instead of relying on
prompt wording alone. It is intentionally small and data-driven: profiles can
be extended without changing the search algorithm.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional


LAW_ALIASES = {
    "GMBH": "GMBHG",
}


@dataclass(frozen=True)
class NormRef:
    """A normalized legal source reference."""

    law: str
    paragraph: str
    reason: str = ""

    @property
    def label(self) -> str:
        return f"{self.law} {self.paragraph}"


@dataclass(frozen=True)
class LegalIssueProfile:
    """Issue-specific retrieval and answer requirements."""

    id: str
    title: str
    rechtsgebiet: str
    trigger_any: tuple[str, ...]
    trigger_all: tuple[str, ...] = ()
    required_norms: tuple[NormRef, ...] = ()
    recommended_norms: tuple[NormRef, ...] = ()
    excluded_norms: tuple[NormRef, ...] = ()
    answer_focus: tuple[str, ...] = ()

    def matches(self, query: str, rechtsgebiet: Optional[str] = None) -> bool:
        q = query.lower()
        if rechtsgebiet and rechtsgebiet != self.rechtsgebiet:
            # Allow strong textual matches even when the classifier is unsure,
            # but do not cross-apply profiles to clearly different domains.
            if self.rechtsgebiet.lower() not in q:
                if not any(t.lower() in q for t in self.trigger_any):
                    return False
        if self.trigger_all and not all(t.lower() in q for t in self.trigger_all):
            return False
        return any(t.lower() in q for t in self.trigger_any)


@dataclass
class LegalRetrievalPlan:
    """Search and answer requirements derived from a query."""

    query: str
    rechtsgebiet: Optional[str] = None
    profiles: list[str] = field(default_factory=list)
    required_norms: list[NormRef] = field(default_factory=list)
    recommended_norms: list[NormRef] = field(default_factory=list)
    excluded_norms: list[NormRef] = field(default_factory=list)
    answer_focus: list[str] = field(default_factory=list)

    @property
    def has_profile(self) -> bool:
        return bool(self.profiles)


@dataclass
class SourceAudit:
    """Audit result for the source set that will be sent to the LLM."""

    accepted_count: int
    rejected: list[dict[str, str]]
    injected: list[str]
    missing_required: list[str]
    notes: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "accepted_count": self.accepted_count,
            "rejected": self.rejected,
            "injected": self.injected,
            "missing_required": self.missing_required,
            "notes": self.notes,
        }


ISSUE_PROFILES: tuple[LegalIssueProfile, ...] = (
    LegalIssueProfile(
        id="arbeitsrecht_ordentliche_kuendigung_arbeitnehmer",
        title="Ordentliche Kündigung eines Arbeitnehmers",
        rechtsgebiet="Arbeitsrecht",
        trigger_any=(
            "ordentliche kündigung",
            "ordentlich kündigen",
            "arbeitnehmer kündigen",
            "kündigung arbeitnehmer",
            "arbeitgeber kündigt",
        ),
        trigger_all=("kündigung",),
        required_norms=(
            NormRef("BGB", "§ 623", "Schriftform der Kündigung"),
            NormRef("BGB", "§ 622", "gesetzliche Kündigungsfristen"),
            NormRef("KSchG", "§ 1", "soziale Rechtfertigung bei Anwendbarkeit des KSchG"),
            NormRef("KSchG", "§ 4", "Dreiwochenfrist für Kündigungsschutzklage"),
            NormRef("KSchG", "§ 23", "betrieblicher Anwendungsbereich des KSchG"),
            NormRef("BetrVG", "§ 102", "Betriebsratsanhörung vor Kündigung"),
        ),
        recommended_norms=(
            NormRef("SGB IX", "§ 168", "Zustimmung des Integrationsamts bei Schwerbehinderung"),
        ),
        excluded_norms=(
            NormRef("BGB", "§ 580a", "Mietrechtliche Kündigungsfrist, nicht Arbeitsverhältnis"),
            NormRef("BetrVG", "§ 103", "Sonderfall außerordentliche Kündigung von Betriebsratsmitgliedern"),
            NormRef("SGB IX", "§ 175", "Erweiterter Beendigungsschutz, nicht die primäre Zustimmungsvorschrift bei ordentlicher Kündigung"),
        ),
        answer_focus=(
            "Schriftform und Zugang prüfen",
            "gesetzliche, vertragliche und tarifliche Kündigungsfrist trennen",
            "KSchG-Anwendbarkeit und Kündigungsgrund prüfen",
            "Betriebsratsanhörung und Sonderkündigungsschutz als Wirksamkeitsrisiken nennen",
            "Kündigungsschutzklagefrist ausdrücklich hervorheben",
        ),
    ),
)


DOMAIN_ANSWER_FOCUS: dict[str, tuple[str, ...]] = {
    "Zivilrecht": (
        "Anspruch entstanden, erloschen und durchsetzbar getrennt prüfen",
        "Vertragsgrundlage, Pflichtverletzung, Vertretenmüssen, Schaden und Fristen sauber trennen",
    ),
    "Arbeitsrecht": (
        "Form, Zugang, Fristen, Beteiligungsrechte und Sonderkündigungsschutz immer prüfen",
        "Tarifvertrag, Arbeitsvertrag und Betriebsvereinbarung als mögliche Abweichungen nennen",
    ),
    "Strafrecht": (
        "Tatbestand, Rechtswidrigkeit und Schuld getrennt prüfen",
        "Strafanzeige, Strafantrag und Verfolgungsverjährung unterscheiden",
    ),
    "Verwaltungsrecht": (
        "Ermächtigungsgrundlage, formelle und materielle Rechtmäßigkeit trennen",
        "Rechtsbehelf und Frist ausdrücklich prüfen",
    ),
    "Insolvenzrecht": (
        "Zahlungsunfähigkeit, Überschuldung, Fristen und Organpflichten trennen",
        "Haftungs- und Strafbarkeitsrisiken nur quellenbasiert nennen",
    ),
}


def normalize_law(law: str) -> str:
    value = (law or "").strip().upper()
    return LAW_ALIASES.get(value, value)


def normalize_paragraph(paragraph: str) -> str:
    text = (paragraph or "").strip()
    m = re.search(r"§+\s*(\d+[a-z]?)", text, flags=re.IGNORECASE)
    if not m:
        return text
    return f"§ {m.group(1)}"


def source_key(source: dict[str, Any]) -> tuple[str, str]:
    return (
        normalize_law(source.get("abkürzung", "") or source.get("gesetz", "")),
        normalize_paragraph(source.get("paragraph", "")),
    )


def norm_key(norm: NormRef) -> tuple[str, str]:
    return (normalize_law(norm.law), normalize_paragraph(norm.paragraph))


def build_retrieval_plan(query: str, rechtsgebiet: Optional[str] = None) -> LegalRetrievalPlan:
    """Build deterministic retrieval requirements for a query."""
    plan = LegalRetrievalPlan(query=query, rechtsgebiet=rechtsgebiet)
    seen_required: set[tuple[str, str]] = set()
    seen_recommended: set[tuple[str, str]] = set()
    seen_excluded: set[tuple[str, str]] = set()

    for profile in ISSUE_PROFILES:
        if not profile.matches(query, rechtsgebiet):
            continue
        plan.profiles.append(profile.id)
        for norm in profile.required_norms:
            key = norm_key(norm)
            if key not in seen_required:
                plan.required_norms.append(norm)
                seen_required.add(key)
        for norm in profile.recommended_norms:
            key = norm_key(norm)
            if key not in seen_recommended:
                plan.recommended_norms.append(norm)
                seen_recommended.add(key)
        for norm in profile.excluded_norms:
            key = norm_key(norm)
            if key not in seen_excluded:
                plan.excluded_norms.append(norm)
                seen_excluded.add(key)
        plan.answer_focus.extend(profile.answer_focus)

    for focus in DOMAIN_ANSWER_FOCUS.get(rechtsgebiet or "", ()):
        if focus not in plan.answer_focus:
            plan.answer_focus.append(focus)

    return plan


def _doc_text(doc: dict[str, Any]) -> str:
    return doc.get("inhalt", "") or doc.get("volltext", "") or doc.get("leitsatz", "")


def _find_exact_norm(searcher: Any, norm: NormRef) -> Optional[dict[str, Any]]:
    indexer = getattr(searcher, "indexer", None)
    documents = getattr(indexer, "documents", None)
    if not documents:
        return None
    wanted = norm_key(norm)
    for idx, doc in enumerate(documents):
        if source_key(doc) != wanted:
            continue
        if not _doc_text(doc).strip():
            continue
        found = dict(doc)
        found["score"] = max(float(found.get("score", 0) or 0), 1.05)
        found["doc_index"] = idx
        found["context_type"] = "required_norm"
        found["quality_reason"] = norm.reason
        if hasattr(indexer, "_doc_label"):
            found["label"] = indexer._doc_label(found)
        found["pid"] = f"{wanted[0]}||{wanted[1]}"
        return found
    return None


def apply_legal_quality(
    query: str,
    results: list[dict[str, Any]],
    searcher: Any,
    rechtsgebiet: Optional[str] = None,
    top_k: int = 10,
) -> tuple[list[dict[str, Any]], LegalRetrievalPlan, SourceAudit]:
    """Inject required sources and remove known false positives.

    The function is deterministic and side-effect free. If no profile matches,
    it still returns domain-level answer focus but leaves sources untouched.
    """
    plan = build_retrieval_plan(query, rechtsgebiet)
    rejected: list[dict[str, str]] = []
    injected: list[str] = []
    notes: list[str] = []

    excluded = {norm_key(n): n for n in plan.excluded_norms}
    required_keys = {norm_key(n) for n in plan.required_norms}
    recommended_keys = {norm_key(n) for n in plan.recommended_norms}
    issue_source_keys = required_keys | recommended_keys
    filtered: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    for source in results:
        key = source_key(source)
        if key in excluded:
            rejected.append({
                "source": f"{key[0]} {key[1]}",
                "reason": excluded[key].reason,
            })
            continue
        if plan.has_profile and source.get("context_type") == "citation_kg" and key not in issue_source_keys:
            rejected.append({
                "source": f"{key[0]} {key[1]}",
                "reason": "KG-Kontext ist für das erkannte Prüfprofil keine Pflicht- oder Sonderquelle",
            })
            continue
        if (
            plan.has_profile
            and plan.rechtsgebiet
            and source.get("rechtsgebiet")
            and source.get("rechtsgebiet") != plan.rechtsgebiet
            and key not in issue_source_keys
        ):
            rejected.append({
                "source": f"{key[0]} {key[1]}",
                "reason": f"Rechtsgebiet {source.get('rechtsgebiet')} passt nicht zum Prüfprofil {plan.rechtsgebiet}",
            })
            continue
        if key in seen:
            continue
        filtered.append(source)
        seen.add(key)

    for norm in plan.required_norms:
        key = norm_key(norm)
        if key in seen:
            continue
        exact = _find_exact_norm(searcher, norm)
        if exact:
            filtered.insert(0, exact)
            seen.add(key)
            injected.append(norm.label)

    for norm in plan.recommended_norms:
        key = norm_key(norm)
        if key in seen:
            continue
        exact = _find_exact_norm(searcher, norm)
        if exact:
            exact["context_type"] = "recommended_norm"
            filtered.append(exact)
            seen.add(key)
            injected.append(norm.label)

    missing_required = [
        norm.label for norm in plan.required_norms if norm_key(norm) not in seen
    ]
    if missing_required:
        notes.append("Pflichtquellen fehlen im Index oder konnten nicht exakt aufgelöst werden.")
    if rejected:
        notes.append("Fachfremde oder profilwidrige Quellen wurden vor der Antwort entfernt.")
    if injected:
        notes.append("Pflichtquellen wurden deterministisch ergänzt.")

    limit = max(top_k, min(len(filtered), top_k + len(injected) + 2))
    audit = SourceAudit(
        accepted_count=min(len(filtered), limit),
        rejected=rejected,
        injected=injected,
        missing_required=missing_required,
        notes=notes,
    )
    return filtered[:limit], plan, audit


def plan_to_dict(plan: LegalRetrievalPlan) -> dict[str, Any]:
    return {
        "query": plan.query,
        "rechtsgebiet": plan.rechtsgebiet,
        "profiles": plan.profiles,
        "required_norms": [n.label for n in plan.required_norms],
        "recommended_norms": [n.label for n in plan.recommended_norms],
        "excluded_norms": [n.label for n in plan.excluded_norms],
        "answer_focus": plan.answer_focus,
    }


def build_answer_requirements(plan_data: Optional[dict[str, Any]], audit_data: Optional[dict[str, Any]]) -> str:
    """Render concise answer requirements for the LLM prompt."""
    if not plan_data:
        return ""
    lines: list[str] = []
    profiles = plan_data.get("profiles") or []
    if profiles:
        lines.append("QUALITÄTSANFORDERUNGEN AUS DER RECHTLICHEN PRÜFPLANUNG:")
        lines.append("- Diese Anforderungen sind verbindlich. Behandle sie in der Antwort sichtbar, nicht nur in der Quellenliste.")
        required = plan_data.get("required_norms") or []
        if required:
            lines.append("- Pflichtquellen, soweit im Kontext vorhanden, ausdrücklich prüfen: " + ", ".join(required))
        recommended = plan_data.get("recommended_norms") or []
        if recommended:
            lines.append("- Mögliche Sonderregeln gesondert als kontextabhängig markieren: " + ", ".join(recommended))
    focus = plan_data.get("answer_focus") or []
    for item in focus:
        lines.append(f"- {item}")
    if audit_data:
        missing = audit_data.get("missing_required") or []
        if missing:
            lines.append("- Falls Pflichtquellen fehlen, transparent sagen, dass die Antwort nicht vollständig abgesichert ist: " + ", ".join(missing))
        rejected = audit_data.get("rejected") or []
        if rejected:
            rejected_labels = ", ".join(r.get("source", "") for r in rejected if r.get("source"))
            lines.append("- Nicht auf ausgeschlossene fachfremde Quellen stützen: " + rejected_labels)
    return "\n".join(lines)
