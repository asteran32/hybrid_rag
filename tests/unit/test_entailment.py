from types import SimpleNamespace

import numpy as np
import pytest

from enterprise_agent.validation import entailment


def test_local_verifier_batches_pairs_and_reads_labels_from_model_configuration(
    monkeypatch,
):
    batches = []

    class Tokenizer:
        def encode_batch(self, pairs):
            batches.append(pairs)
            return [
                SimpleNamespace(ids=[1, 2, 3], attention_mask=[1, 1, 1]) for _ in pairs
            ]

    class Session:
        def get_inputs(self):
            return [
                SimpleNamespace(name="input_ids"),
                SimpleNamespace(name="attention_mask"),
            ]

        def run(self, outputs, inputs):
            assert set(inputs) == {"input_ids", "attention_mask"}
            assert inputs["input_ids"].dtype == np.int64
            return [np.tile([0.0, 3.0, -2.0], (len(inputs["input_ids"]), 1))]

    monkeypatch.setattr(
        entailment,
        "_load",
        lambda: (Tokenizer(), Session(), {"entailment": 1, "contradiction": 2}),
    )
    pairs = [(f"source {n}", f"claim {n}") for n in range(9)]
    results = entailment.classify_local(pairs)
    assert batches == [pairs[:8], pairs[8:]]
    assert len(results) == 9 and all(
        r.entailment > 0.94 and r.contradiction < 0.01 for r in results
    )


def test_unavailable_verifier_surfaces_an_operational_error(monkeypatch):
    def missing():
        raise OSError("private filesystem detail")

    monkeypatch.setattr(entailment, "_load", missing)
    with pytest.raises(entailment.VerificationUnavailable, match="cache") as exc:
        entailment.classify_local([("a", "b")])
    assert "private filesystem detail" not in str(exc.value)
