from collections.abc import Iterable


def document_metrics(
    retrieved: Iterable[str], expected: set[str]
) -> dict[str, float] | None:
    if not expected:
        return None
    documents = list(dict.fromkeys(retrieved))
    first = next(
        (rank for rank, doc in enumerate(documents, 1) if doc in expected), None
    )
    return {
        **{
            f"recall@{k}": len(expected.intersection(documents[:k])) / len(expected)
            for k in (1, 5, 10)
        },
        "mrr": 1 / first if first else 0.0,
    }
