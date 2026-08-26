"""Conclusions about a run, written by a model and authorized by the data.

``compass compare`` answers precisely and says nothing. It will tell you the
pass-rate difference, the confidence interval, which cases flipped, and what
effect size this sample could have resolved — and then it stops, because the
next sentence ("the failures cluster in the retrieval cases") is a reading, not
a measurement. Someone has to write that sentence, and today it is always the
person squinting at the table.

A model can write it. The problem has never been fluency; it is that a fluent
wrong sentence about an eval is worse than no sentence, because it reads
exactly like a right one. So this module does not ask the model to be careful.
It lets the model write whatever it likes, and then **checks every claim
against the run and throws away the ones the data does not support**:

    judge = InsightsJudge(model="gpt-5.6-sol")
    report = await judge.run(doc)
    for claim in report.claims:      # only the survivors
        print(claim.severity, claim.title, claim.case_ids)
    report.dropped                   # and what did not survive, with reasons

**The model chooses the words; the data decides whether they may be said.**

Grounding is per claim, and a claim that fails any check is dropped whole
rather than softened:

- **The type must be one this run can support.** A ``regression`` claim needs a
  comparison to have been supplied; without one there is no such thing as a
  case that got worse, and the claim goes.
- **Every cited case must actually satisfy the type's predicate.** A
  ``failure_pattern`` citing a case that passed is not a weaker insight, it is
  a false one.
- **A harness error is not an agent failure.** Compass keeps crashed cases out
  of every denominator; a ``failure_pattern`` that cites one is dropped, and
  the judge has a separate ``harness_error`` type for saying so honestly. This
  is the check most worth having: "3 of 15 failed" and "3 of 15 never ran" look
  identical in prose and mean opposite things.
- **The cases cited must be the cases discussed.** Every id in ``case_ids`` has
  to appear in the claim's own text, and no id outside that list may. Citing
  ``neg_chart`` while describing what happened to ``pos_implicit`` is how a
  claim ends up unfalsifiable — the reader follows the citation and finds
  something that does not match, or worse, does not follow it.
- **Severity has to agree with the type.** A regression reported as ``pass`` is
  a claim about the data that the data contradicts.
- **A claim about absence must cite nothing.** ``coverage`` says the suite has
  a blind spot, which is exactly the claim no case can evidence; naming one
  turns it into a claim about that case, and it is checked as one.

What survives is not "probably true" — it is "consistent with what this run
measured", which is a narrower and much more useful guarantee. The rest is
reported in :attr:`InsightsReport.dropped` with its reason, because a judge
that silently discards half its output is a judge you cannot calibrate.

**What this does not do.** It never grounds a claim against a *number* the run
did not turn into a verdict. There is no "these cases were expensive" type: if
a ``cost_budget`` grader ran and failed, that is a ``process_defect`` and
checkable; if none ran, the run did not measure cost as a defect and a judge
eyeballing the column is guessing. Compass grades what it graded.

And it is a judge, so it is not deterministic. Grounding makes the output
*safe*, not *stable*: two runs over the same document can surface different
true claims. Read it as a reviewer's notes, never as a metric.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from compass.llm import structured_completion

if TYPE_CHECKING:
    from compass.report.compare import ComparisonReport

logger = logging.getLogger(__name__)

#: Default judge. Overridden per call; a smaller model is a fine trade here
#: because grounding catches what a weaker reader gets wrong.
DEFAULT_MODEL = "gpt-5.6-sol"

#: How many claims survive into the report, at most. A reviewer's note that
#: runs to thirty bullet points is a table again.
DEFAULT_MAX_CLAIMS = 6

#: How many cases are described to the judge. Failures, flips and tagged cases
#: go first; what does not fit is counted, not silently dropped.
DEFAULT_MAX_CASES = 80

_MAX_TITLE_CHARS = 120
_MAX_MESSAGE_CHARS = 600

_SEVERITIES = ("pass", "warn", "fail")

#: Severities each claim type may carry. A regression reported as `pass`
#: contradicts the thing it is reporting.
_ALLOWED_SEVERITY: dict[str, tuple[str, ...]] = {
    "failure_pattern": ("warn", "fail"),
    "regression": ("warn", "fail"),
    "improvement": ("pass", "warn"),
    "process_defect": ("warn", "fail"),
    "harness_error": ("warn", "fail"),
    "coverage": _SEVERITIES,
}

#: Types that only exist when two runs were compared.
_NEEDS_COMPARISON = frozenset({"regression", "improvement"})

#: The type that asserts an absence, and therefore may cite nothing.
_CITES_NOTHING = "coverage"

CLAIM_TYPES: tuple[str, ...] = tuple(_ALLOWED_SEVERITY)


# ---------------------------------------------------------------------------
# Facts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CaseFacts:
    """What one case actually did — the only thing a claim may lean on."""

    case_id: str
    scenario: str = ""
    category: str = ""
    status: str = ""
    passed: bool = False
    score: float | None = None
    #: The harness crashed. Not an agent failure, and excluded from every
    #: denominator Compass reports.
    errored: bool = False
    #: Graders that measured and did not pass.
    failed_graders: tuple[str, ...] = ()
    #: A failing grader whose scope covers the trajectory, i.e. the run's
    #: *process* was judged defective rather than its answer.
    process_failed: bool = False
    outcome_failed: bool = False
    failure_tags: tuple[str, ...] = ()
    observed_tags: tuple[str, ...] = ()
    #: "" | "improved" | "regressed", from a paired comparison.
    flip: str = ""

    def to_prompt_dict(self) -> dict[str, Any]:
        """The compact form the judge reads. Omits what it cannot cite on."""
        row: dict[str, Any] = {"case_id": self.case_id, "status": self.status}
        if self.score is not None:
            row["score"] = round(self.score, 3)
        for key, value in (
            ("scenario", self.scenario),
            ("category", self.category),
            ("flip", self.flip),
        ):
            if value:
                row[key] = value
        if self.failed_graders:
            row["failed_graders"] = list(self.failed_graders)
        if self.process_failed:
            row["process_failed"] = True
        if self.failure_tags:
            row["failure_tags"] = list(self.failure_tags)
        if self.observed_tags:
            row["observed_tags"] = list(self.observed_tags)
        return row


def collect_facts(
    doc: Mapping[str, Any], comparison: ComparisonReport | None = None
) -> list[CaseFacts]:
    """Read a run document into per-case facts, in document order.

    *doc* is what ``collect_run`` / ``collect_run_payload`` produce — the same
    shape the site and the HTML report render, so a claim is checked against
    exactly what a reader would see.
    """
    from compass.report.site import iter_cases

    flips: dict[str, str] = {}
    if comparison is not None:
        flips = {f.case_id: "improved" for f in comparison.improved}
        flips.update({f.case_id: "regressed" for f in comparison.regressed})

    facts: list[CaseFacts] = []
    for case in iter_cases(doc):
        case_id = str(case.get("case_id") or case.get("task_id") or "")
        if not case_id:
            continue
        graders = [g for g in case.get("evaluator_results") or [] if isinstance(g, dict)]
        measured = [
            g for g in graders if not g.get("skipped") and g.get("score") is not None
        ]
        failed = [g for g in measured if not g.get("passed")]
        failure_tags = {
            str(t) for g in graders for t in (g.get("failure_tags") or []) if t
        }
        status = str(case.get("status") or "")
        facts.append(
            CaseFacts(
                case_id=case_id,
                scenario=str(case.get("scenario") or ""),
                category=str(case.get("category") or ""),
                status=status,
                passed=bool(case.get("passed")),
                score=_opt_float(case.get("overall_score")),
                errored=status == "error",
                failed_graders=tuple(str(g.get("name") or "?") for g in failed),
                process_failed=any(
                    str(g.get("grader_scope") or "outcome") in ("transcript", "both")
                    for g in failed
                ),
                outcome_failed=any(
                    str(g.get("grader_scope") or "outcome") in ("outcome", "both")
                    for g in failed
                ),
                failure_tags=tuple(sorted(failure_tags)),
                observed_tags=tuple(
                    str(t) for t in (case.get("observed_tags") or []) if t
                ),
                flip=flips.get(case_id, ""),
            )
        )
    return facts


# ---------------------------------------------------------------------------
# Claims
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Claim:
    """One conclusion the data was willing to authorize."""

    claim_type: str
    severity: str
    title: str
    message: str
    case_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_type": self.claim_type,
            "severity": self.severity,
            "title": self.title,
            "message": self.message,
            "case_ids": list(self.case_ids),
        }


@dataclass(frozen=True)
class DroppedClaim:
    """A claim the data did not support, and why it did not."""

    reason: str
    claim_type: str = ""
    title: str = ""
    case_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "reason": self.reason,
            "claim_type": self.claim_type,
            "title": self.title,
            "case_ids": list(self.case_ids),
        }


@dataclass
class InsightsReport:
    """Everything the judge produced, kept and discarded alike."""

    claims: list[Claim] = field(default_factory=list)
    dropped: list[DroppedClaim] = field(default_factory=list)
    model: str = ""
    cases_considered: int = 0
    cases_elided: int = 0
    #: Set when the judge could not run at all (no key, no package, a provider
    #: error). Distinct from "ran and had nothing to say".
    error: str = ""

    @property
    def grounded_rate(self) -> float:
        """Share of the judge's claims that survived. A calibration signal:
        a judge dropping most of what it writes is being read the wrong data."""
        total = len(self.claims) + len(self.dropped)
        return len(self.claims) / total if total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "claims": [c.to_dict() for c in self.claims],
            "dropped": [d.to_dict() for d in self.dropped],
            "cases_considered": self.cases_considered,
            "cases_elided": self.cases_elided,
            "grounded_rate": self.grounded_rate,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Grounding
# ---------------------------------------------------------------------------


def _holds(claim_type: str, facts: CaseFacts) -> bool:
    """Whether one case is admissible evidence for a claim of this type."""
    if claim_type == "failure_pattern":
        # A crashed case is not the agent getting it wrong — it is the harness
        # not getting far enough to ask. `harness_error` is where it belongs.
        return not facts.passed and not facts.errored
    if claim_type == "regression":
        return facts.flip == "regressed"
    if claim_type == "improvement":
        return facts.flip == "improved"
    if claim_type == "process_defect":
        return facts.process_failed or bool(facts.failure_tags)
    if claim_type == "harness_error":
        return facts.errored
    return False


def ground(
    claims: Sequence[Mapping[str, Any]],
    facts: Sequence[CaseFacts],
    *,
    has_comparison: bool = False,
    max_claims: int = DEFAULT_MAX_CLAIMS,
) -> tuple[list[Claim], list[DroppedClaim]]:
    """Split the judge's claims into the ones the data supports and the rest.

    Every check is a statement about the run, never about style: a claim is
    dropped because the cases it names do not do what it says they do, not
    because of how it says it.
    """
    by_id = {f.case_id: f for f in facts}
    known = set(by_id)
    kept: list[Claim] = []
    dropped: list[DroppedClaim] = []
    seen: set[tuple[str, frozenset[str]]] = set()

    for raw in claims:
        if not isinstance(raw, Mapping):
            dropped.append(DroppedClaim(reason="not an object"))
            continue

        claim_type = str(raw.get("claim_type") or "")
        title = str(raw.get("title") or "").strip()
        message = str(raw.get("message") or "").strip()
        severity = str(raw.get("severity") or "")
        cited = tuple(dict.fromkeys(str(c) for c in raw.get("case_ids") or [] if c))
        stub = DroppedClaim(reason="", claim_type=claim_type, title=title, case_ids=cited)

        def drop(reason: str, stub: DroppedClaim = stub) -> None:
            dropped.append(
                DroppedClaim(
                    reason=reason,
                    claim_type=stub.claim_type,
                    title=stub.title,
                    case_ids=stub.case_ids,
                )
            )

        if claim_type not in _ALLOWED_SEVERITY:
            drop(f"unknown claim_type {claim_type!r}")
            continue
        if not title or not message:
            drop("a claim with no title or no message says nothing")
            continue
        if len(title) > _MAX_TITLE_CHARS or len(message) > _MAX_MESSAGE_CHARS:
            drop("over length")
            continue
        if severity not in _ALLOWED_SEVERITY[claim_type]:
            drop(
                f"severity {severity!r} contradicts a {claim_type} claim "
                f"(allowed: {', '.join(_ALLOWED_SEVERITY[claim_type])})"
            )
            continue
        if claim_type in _NEEDS_COMPARISON and not has_comparison:
            drop(f"{claim_type} needs two runs compared; this report has one")
            continue

        text = f"{title}\n{message}"
        named = _ids_in_text(text, known)

        if claim_type == _CITES_NOTHING:
            if cited or named:
                drop("a coverage claim names cases; it is a claim about them")
                continue
        else:
            if not cited:
                drop(f"a {claim_type} claim cites no case to check it against")
                continue
            unknown = [c for c in cited if c not in known]
            if unknown:
                drop(f"cites case(s) not in this run: {', '.join(sorted(unknown))}")
                continue
            if named != set(cited):
                drop(
                    "the cases cited are not the cases discussed "
                    f"(cited {sorted(cited)}, named in the text {sorted(named)})"
                )
                continue
            failing = [c for c in cited if not _holds(claim_type, by_id[c])]
            if failing:
                drop(
                    f"{', '.join(sorted(failing))} does not support a "
                    f"{claim_type} claim: {_why_not(claim_type, by_id[failing[0]])}"
                )
                continue

        key = (claim_type, frozenset(cited))
        if key in seen:
            drop("duplicate of an earlier claim")
            continue
        seen.add(key)

        if len(kept) >= max_claims:
            drop(f"over the {max_claims}-claim limit")
            continue
        kept.append(Claim(claim_type, severity, title, message, cited))

    return kept, dropped


def _why_not(claim_type: str, facts: CaseFacts) -> str:
    """Why one cited case is not admissible — the sentence a reader needs."""
    if claim_type == "failure_pattern":
        if facts.errored:
            return "it errored in the harness, which is not the agent failing"
        return "it passed"
    if claim_type in ("regression", "improvement"):
        return f"its verdict did not flip ({facts.flip or 'unchanged'})"
    if claim_type == "process_defect":
        return "no transcript-scope grader failed on it and it carries no failure tag"
    if claim_type == "harness_error":
        return f"it has status {facts.status or 'unknown'!r}, not error"
    return "it does not match this claim type"


def _ids_in_text(text: str, known: set[str]) -> set[str]:
    """Case ids named in *text*, matched whole.

    Longest id first, consuming the text as it goes. Two failures this avoids,
    and they pull in opposite directions: matching each id independently finds
    ``c1`` inside ``c10``, while widening the boundary to stop that also stops
    ``c1`` being found at the end of a sentence — and a claim whose last word
    is the case it cites is the normal shape, not the exotic one. Taking the
    longest match first and masking it settles both: ``c10`` claims its span
    before ``c1`` can look inside it, and ``c1.`` still reads as ``c1``.
    """
    found: set[str] = set()
    remaining = text
    for case_id in sorted(known, key=len, reverse=True):
        # A word character or a hyphen either side means this is part of a
        # longer name; punctuation and whitespace are ordinary prose.
        pattern = rf"(?<![A-Za-z0-9_\-]){re.escape(case_id)}(?![A-Za-z0-9_\-])"
        masked, hits = re.subn(pattern, lambda m: "\0" * len(m.group(0)), remaining)
        if hits:
            found.add(case_id)
            remaining = masked
    return found


# ---------------------------------------------------------------------------
# The judge
# ---------------------------------------------------------------------------


_SYSTEM = (
    "You review AI-agent evaluation runs. You write short, concrete "
    "conclusions a maintainer can act on. Every claim you make is checked "
    "against the run's data and discarded if it does not hold, so a cautious "
    "true claim is worth more than a sweeping one. Reply only with JSON "
    "matching the schema."
)


def _schema(types: Sequence[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "claims": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "claim_type": {"type": "string", "enum": list(types)},
                        "severity": {"type": "string", "enum": list(_SEVERITIES)},
                        "title": {"type": "string"},
                        "message": {"type": "string"},
                        "case_ids": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": [
                        "claim_type",
                        "severity",
                        "title",
                        "message",
                        "case_ids",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["claims"],
        "additionalProperties": False,
    }


def _rank(facts: CaseFacts) -> tuple[int, str]:
    """Sort key for which cases the judge gets to see when they do not all fit.

    Flips first (they are the story of a comparison), then harness errors,
    then failures, then anything carrying a tag. Passing, untagged cases are
    context and go last.
    """
    if facts.flip:
        return (0, facts.case_id)
    if facts.errored:
        return (1, facts.case_id)
    if not facts.passed:
        return (2, facts.case_id)
    if facts.failure_tags or facts.observed_tags:
        return (3, facts.case_id)
    return (4, facts.case_id)


def build_prompt(
    facts: Sequence[CaseFacts],
    *,
    summary: Mapping[str, Any],
    comparison: ComparisonReport | None = None,
    max_claims: int = DEFAULT_MAX_CLAIMS,
) -> str:
    """The judge's whole view of the run: a summary, the cases, and the rules."""
    payload: dict[str, Any] = {
        "run": dict(summary),
        "cases": [f.to_prompt_dict() for f in facts],
    }
    if comparison is not None:
        payload["comparison"] = {
            "label_a": comparison.label_a,
            "label_b": comparison.label_b,
            "n_paired": comparison.n_paired,
            "verdict": comparison.verdict,
            "pass_rate_a": comparison.pass_rate_a,
            "pass_rate_b": comparison.pass_rate_b,
            "improved": [f.case_id for f in comparison.improved],
            "regressed": [f.case_id for f in comparison.regressed],
        }

    rules = [
        "failure_pattern — cases that failed and share a cause. Never cite a "
        "case whose status is 'error': the harness crashed, the agent was not "
        "asked. Use harness_error for those.",
        "process_defect — cases where the *way* the agent worked was judged "
        "defective (a transcript-scope grader failed, or a failure tag was "
        "raised), whatever the final answer was.",
        "harness_error — cases that never ran to completion. Say how many, and "
        "that the pass rate excludes them.",
        "coverage — a gap in what the suite tests. This one must cite NO "
        "cases; it is a claim about what is absent.",
    ]
    if comparison is not None:
        rules[:0] = [
            "regression — cases whose verdict went from pass to fail between "
            "the two runs. Only cases in comparison.regressed qualify.",
            "improvement — cases whose verdict went from fail to pass. Only "
            "cases in comparison.improved qualify.",
        ]

    return "\n".join(
        [
            "Here is one evaluation run, as Compass measured it.",
            "",
            json.dumps(payload, ensure_ascii=False, indent=1),
            "",
            f"Write at most {max_claims} claims. Each has a claim_type from:",
            "",
            *(f"- {rule}" for rule in rules),
            "",
            "Rules your output is checked against — a claim that breaks one is "
            "discarded, not corrected:",
            "",
            "1. Every case_id you list must appear verbatim in your own title "
            "or message, and you must not name a case you did not list. Cite "
            "what you discuss; discuss what you cite.",
            "2. Every case you cite must actually satisfy the claim's type. "
            "Check the case's row above before citing it.",
            "3. severity must agree with the claim: failure_pattern, "
            "regression, process_defect and harness_error are 'warn' or "
            "'fail'; improvement is 'pass' or 'warn'.",
            "4. Only these case ids exist. Do not invent one, and do not refer "
            "to a case by paraphrase.",
            "",
            "Say less rather than more. An empty claims list is a valid answer "
            "for a run with nothing notable in it.",
        ]
    )


class InsightsJudge:
    """Runs the model, then lets the data decide what it was allowed to say.

    Args:
        model: Model id. Defaults to :data:`DEFAULT_MODEL`.
        provider: ``"openai"`` or ``"anthropic"``.
        max_claims: Cap on surviving claims.
        max_cases: Cap on cases described to the judge. What does not fit is
            counted into ``cases_elided`` — a claim can only cite what it saw,
            so a truncated view narrows the output rather than corrupting it.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        provider: str = "openai",
        max_claims: int = DEFAULT_MAX_CLAIMS,
        max_cases: int = DEFAULT_MAX_CASES,
        call: Callable[[str, dict[str, Any]], Any] | None = None,
    ) -> None:
        self.model = model
        self.provider = provider
        self.max_claims = max_claims
        self.max_cases = max_cases
        self._call = call

    async def run(
        self,
        doc: Mapping[str, Any],
        comparison: ComparisonReport | None = None,
    ) -> InsightsReport:
        """Judge one run document, optionally in light of a paired comparison."""
        facts = collect_facts(doc, comparison)
        shown = sorted(facts, key=_rank)[: self.max_cases]
        report = InsightsReport(
            model=self.model,
            cases_considered=len(shown),
            cases_elided=max(0, len(facts) - len(shown)),
        )
        if not shown:
            report.error = "no cases in this run — nothing to draw a conclusion from"
            return report

        types = [
            t for t in CLAIM_TYPES if comparison is not None or t not in _NEEDS_COMPARISON
        ]
        prompt = build_prompt(
            shown,
            summary=_summary(doc),
            comparison=comparison,
            max_claims=self.max_claims,
        )
        try:
            payload = await self._call_llm_structured(prompt, _schema(types))
        except Exception as exc:  # noqa: BLE001 — any provider failure reads the same
            # A judge that cannot run must not look like a judge that found
            # nothing: the caller has to be able to tell those apart.
            report.error = f"{type(exc).__name__}: {exc}"
            logger.warning("insights judge did not run: %s", exc)
            return report

        raw = payload.get("claims") if isinstance(payload, Mapping) else None
        report.claims, report.dropped = ground(
            raw if isinstance(raw, list) else [],
            shown,
            has_comparison=comparison is not None,
            max_claims=self.max_claims,
        )
        return report

    async def _call_llm_structured(
        self, prompt: str, schema: dict[str, Any]
    ) -> dict[str, Any]:
        """The seam. Patched in tests; a live call otherwise."""
        if self._call is not None:
            result = await self._call(prompt, schema)
            return result if isinstance(result, dict) else {}
        return await structured_completion(
            prompt,
            schema=schema,
            model=self.model,
            provider=self.provider,
            system=_SYSTEM,
            schema_name="insights",
        )


def _summary(doc: Mapping[str, Any]) -> dict[str, Any]:
    """The run-level numbers the judge is allowed to reason from."""
    keys = (
        "total_cases",
        "evaluated_cases",
        "passed_cases",
        "failed_cases",
        "error_cases",
        "pass_rate",
        "average_score",
    )
    summary = {k: doc[k] for k in keys if k in doc}
    if doc.get("run"):
        run = doc["run"]
        if isinstance(run, Mapping) and run.get("name"):
            summary["name"] = run["name"]
    return summary


def _opt_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
