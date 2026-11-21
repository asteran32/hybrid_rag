import re
from decimal import Decimal
from unicodedata import normalize

from enterprise_agent.generation.contracts import AnswerDraft, DraftConflict, Limitation
from enterprise_agent.validation.citations import Passage
from enterprise_agent.validation.entailment import Classifier

# Accept support only when it outweighs the other two NLI classes combined.
ENTAILMENT = 0.50
CONTRADICTION = 0.90


def _numeric_text(text: str) -> str:
    return re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", normalize("NFKC", text)).replace(
        "−", "-"
    )


def _numbers(text: str) -> set[Decimal]:
    return {
        Decimal(value)
        for value in re.findall(
            r"(?<!\w)[+-]?\d+(?:\.\d+)?|\d+(?:\.\d+)?", _numeric_text(text)
        )
    }


def _anchors_match(statement: str, evidence: str) -> bool:
    # NLI can overlook a changed number or an invented acronym in translated prose.
    acronyms = re.findall(
        r"\b[A-Z][A-Z0-9]{1,}\b", normalize("NFKC", statement), re.ASCII
    )
    words = set(re.findall(r"[a-z0-9]+", normalize("NFKC", evidence).casefold()))
    return _numbers(statement) <= _numbers(evidence) and all(
        a.casefold() in words for a in acronyms
    )


def _numeric_assertion(text: str) -> tuple[str, Decimal] | None:
    # Require identical wording, units and qualifiers around an explicit numeric value.
    normalized = " ".join(_numeric_text(text).casefold().split())
    matches = list(
        re.finditer(
            r"(?:\b(?:is|are|was|were|equals)\s+|[:=]\s*)[$€£]?(-?\d+(?:\.\d+)?)",
            normalized,
        )
    )
    if len(matches) != 1:
        return None
    match = matches[0]
    return normalized[: match.start(1)] + "#" + normalized[match.end(1) :], Decimal(
        match[1]
    )


def review_support(
    draft: AnswerDraft, passages: dict[str, Passage], classify: Classifier
) -> tuple[AnswerDraft, list[Limitation]]:
    """Check entailment of cited evidence and search the same context for counterevidence."""
    quotes = {key: passage.text for key, passage in passages.items()}
    premises = {
        key: f"{p.chunk.title}. {p.text}" if p.chunk.title else p.text
        for key, p in passages.items()
    }
    assertions = {key: _numeric_assertion(quote) for key, quote in quotes.items()}

    def conflicting_values(first: str, second: str) -> bool:
        a, b = assertions[first], assertions[second]
        return bool(a and b and a[0] == b[0] and a[1] != b[1])

    pairs: dict[tuple[str, str], None] = {}
    candidates = []
    kept = []
    conflicts = []
    issues: list[Limitation] = []
    for claim in draft.claims if draft.coverage != "insufficient" else []:
        labels = list(dict.fromkeys(ref.root for ref in claim.citations))
        if not labels or any(label not in passages for label in labels):
            kept.append(claim)  # The citation validator records invalid references.
            continue
        joined = "\n".join(premises[label] for label in labels)
        if not _anchors_match(claim.text, " ".join(quotes[label] for label in labels)):
            issues.append("unsupported_claim")
            continue
        candidates.append((claim, labels, joined))
        pairs.update(((text, claim.text), None) for text in premises.values())
        if len(labels) > 1:
            pairs[joined, claim.text] = None
    for pair in draft.conflicts:
        left, right = quotes.get(pair.left.root), quotes.get(pair.right.root)
        if left is not None and right is not None:
            pairs[left, right] = pairs[right, left] = None

    scores = dict(zip(pairs, classify(list(pairs)), strict=True)) if pairs else {}
    uncertain = set()
    for pair in draft.conflicts:
        left, right = quotes.get(pair.left.root), quotes.get(pair.right.root)
        if left is None or right is None or left == right:
            conflicts.append(pair)  # Let the exact reference checks handle it.
        elif (
            conflicting_values(pair.left.root, pair.right.root)
            or max(scores[left, right].contradiction, scores[right, left].contradiction)
            >= CONTRADICTION
        ):
            conflicts.append(pair)
        else:
            uncertain.update([pair.left.root, pair.right.root])
            issues.append("unverified_conflict")

    seen = {frozenset((pair.left.root, pair.right.root)) for pair in conflicts}
    for claim, labels, joined in candidates:
        support = [
            label
            for label in labels
            if _anchors_match(claim.text, quotes[label])
            and scores[premises[label], claim.text].entailment > ENTAILMENT
        ]
        if (
            not support
            and len(labels) > 1
            and scores[joined, claim.text].entailment > ENTAILMENT
        ):
            support = labels
        if not support:
            issues.append("unsupported_claim")
            continue
        if uncertain.intersection(support):
            continue
        kept.append(
            claim.model_copy(
                update={
                    "citations": [ref for ref in claim.citations if ref.root in support]
                }
            )
        )
        for label, text in premises.items():
            if label in support:
                continue
            opposing = next(
                (ref for ref in support if conflicting_values(ref, label)), None
            )
            if (
                opposing is None
                and scores[text, claim.text].contradiction < CONTRADICTION
            ):
                continue
            source = opposing or support[0]
            key = frozenset((source, label))
            if key not in seen:
                conflicts.append(DraftConflict(left=source, right=label))
                seen.add(key)
    return draft.model_copy(update={"claims": kept, "conflicts": conflicts}), list(
        dict.fromkeys(issues)
    )
