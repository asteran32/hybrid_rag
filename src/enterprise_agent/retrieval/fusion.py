from collections import defaultdict

from enterprise_agent.models import SearchHit


def reciprocal_rank_fusion(
    rankings: dict[str, list[SearchHit]], k: int, constant: int = 60
) -> list[SearchHit]:
    if constant < 0 or k < 1:
        raise ValueError("RRF constant must be nonnegative and k must be positive")
    hits: dict[str, SearchHit] = {}
    ranks: dict[str, dict[str, int]] = defaultdict(dict)
    for method, group in rankings.items():
        for rank, hit in enumerate(group, 1):
            hits.setdefault(hit.chunk_id, hit)
            ranks[hit.chunk_id].setdefault(method, rank)
    scores = {
        cid: sum(1 / (constant + r) for r in methods.values())
        for cid, methods in ranks.items()
    }
    ordered = sorted(hits, key=lambda cid: (-scores[cid], cid))
    return [
        hits[cid].model_copy(
            update={"rank": pos, "score": scores[cid], "component_ranks": ranks[cid]}
        )
        for pos, cid in enumerate(ordered[:k], 1)
    ]
