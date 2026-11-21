import pytest

from enterprise_agent.generation.contracts import AnswerDraft
from enterprise_agent.models import EvidenceChunk
from enterprise_agent.validation.citations import Passage
from enterprise_agent.validation.entailment import Relation
from enterprise_agent.validation.support import review_support

YES, NO, UNKNOWN = Relation(0.99, 0.0), Relation(0.01, 0.96), Relation(0.1, 0.1)


def review(texts, claims, scores, conflicts=()):
    passages = {}
    for label, text in texts.items():
        chunk = EvidenceChunk(
            document_id="dsid_" + label,
            chunk_id=label,
            content=text,
            start_offset=0,
            end_offset=len(text),
            position=0,
            title=None,
            source_type="linear",
            source_path=label,
        )
        passages[label] = Passage(label, chunk, text, 0)
    draft = AnswerDraft(coverage="complete", claims=claims, conflicts=list(conflicts))

    def classify(pairs):
        assert len(set(pairs)) == len(pairs)
        return [scores.get(pair, UNKNOWN) for pair in pairs]

    return review_support(draft, passages, classify)


def fact(text, *labels):
    return {"kind": "fact", "text": text, "citations": list(labels)}


def test_real_quote_does_not_support_fabricated_price_or_unrelated_owner():
    quote = "Archive policy handbook."
    false_price = "The renewal fee is $100."
    invented_owner = "Lee owns archiving."
    output, issues = review(
        {"A": quote},
        [fact(false_price, "A"), fact(invented_owner, "A")],
        {(quote, false_price): YES},
    )
    assert output.claims == [] and issues == ["unsupported_claim"]


def test_only_entailing_citations_survive_and_uncited_contradiction_is_retained():
    texts = {
        "A": "Policy owner is Jo.",
        "B": "The café serves fruit.",
        "C": "Lee is the sole policy owner.",
    }
    text = "Jo owns the policy."
    output, issues = review(
        texts, [fact(text, "A", "B")], {(texts["A"], text): YES, (texts["C"], text): NO}
    )
    assert not issues
    assert [ref.root for ref in output.claims[0].citations] == ["A"]
    assert [(pair.left.root, pair.right.root) for pair in output.conflicts] == [
        ("A", "C")
    ]


def test_unseen_acronym_in_translated_fact_is_rejected_even_with_high_nli_score():
    quote = "오리온 자료실의 담당자는 김지수입니다."
    invented = "김지수 is the manager of OIC."
    out, issues = review({"A": quote}, [fact(invented, "A")], {(quote, invented): YES})
    assert not out.claims and issues == ["unsupported_claim"]


def test_joint_support_and_uncertain_model_reported_disagreement():
    texts = {
        "A": "Parcel B moves on Monday.",
        "B": "Jo receives parcel B.",
        "C": "There are seats.",
    }
    claim = "Jo can expect parcel B on Monday."
    joint = "\n".join([texts["A"], texts["B"]])
    output, issues = review(
        texts,
        [{"kind": "inference", "text": claim, "citations": ["A", "B"]}],
        {(joint, claim): YES},
    )
    assert not issues and [ref.root for ref in output.claims[0].citations] == ["A", "B"]
    output, issues = review(
        texts,
        [fact(claim, "A")],
        {(texts["A"], claim): YES},
        [{"left": "A", "right": "C"}],
    )
    assert (
        not output.claims and not output.conflicts and issues == ["unverified_conflict"]
    )


@pytest.mark.parametrize(
    ("other", "conflict"),
    [
        ("Room capacity is 20 people.", True),
        ("Room capacity was 20 people.", False),
        ("In 2024 room capacity is 20 people.", False),
        ("Room capacity is 12 people.", False),
        ("Room capacity is 12.0 people.", False),
    ],
)
def test_numeric_conflicts_require_matching_subject_units_and_qualifiers(
    other, conflict
):
    original = "Room capacity is 12 people."
    out, _ = review(
        {"A": original, "B": other}, [fact(original, "A")], {(original, original): YES}
    )
    assert bool(out.conflicts) is conflict
