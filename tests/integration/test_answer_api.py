from contextlib import nullcontext

import pytest

from enterprise_agent.generation import runtime
from enterprise_agent.generation.chat import GenerationUnavailable
from enterprise_agent.generation.contracts import Generation, GenerationInfo
from enterprise_agent.ingestion.pipeline import ingest
from enterprise_agent.models import ChunkSettings

pytestmark = pytest.mark.integration


def test_api_uses_trusted_scope_and_fails_closed_on_invalid_input_or_service_failure(
    db, documents, monkeypatch, classify_fixture
):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from enterprise_agent import api

    ingest(db, documents, ChunkSettings(), limit=2, batch_size=2)
    monkeypatch.setattr(api, "connect", lambda: nullcontext(db))
    monkeypatch.delenv("ANSWER_ACCESS_SCOPE", raising=False)
    question = {"question": "business travel receipts", "method": "keyword"}
    with TestClient(api.app) as client:
        denied = client.post("/answer", json=question)
        assert denied.status_code == 200 and denied.json()["generation"] is None
        for extra in [
            {"scope": {"source_types": ["slack"]}},
            {"question": "  "},
            {"top_k": 21},
            {"expected_doc_ids": ["dsid_one"]},
            {"gold_answer": "never accepted"},
        ]:
            assert client.post("/answer", json={**question, **extra}).status_code == 422
        monkeypatch.setenv("ANSWER_ACCESS_SCOPE", '{"source_types":["slack"]}')
        assert (
            client.post("/answer", json={**question, "source_type": "gmail"}).json()
            == denied.json()
        )

        def supported(query, passages):
            assert passages == {"E1.1": documents[0].content}
            return Generation(
                outcome={
                    "coverage": "complete",
                    "claims": [
                        {
                            "kind": "fact",
                            "text": documents[0].content,
                            "citations": ["E1.1"],
                        }
                    ],
                    "conflicts": [],
                },
                info=GenerationInfo(
                    provider="test", model="fixture", prompt_version="test"
                ),
            )

        monkeypatch.setattr(runtime, "generate", supported)
        monkeypatch.setattr(runtime, "classify_local", classify_fixture)
        answer = client.post("/answer", json=question)
        assert answer.status_code == 200 and answer.json()["status"] == "answered"
        assert answer.json()["claims"][0]["citations"][0]["document_id"] == "dsid_one"
        assert answer.json()["answer"] == documents[0].content

        def unavailable(*_):
            raise GenerationUnavailable("sensitive provider details")

        monkeypatch.setattr(runtime, "generate", unavailable)
        failure = client.post("/answer", json=question)
        assert failure.status_code == 503
        assert failure.json() == {"detail": "Answer service unavailable"}
