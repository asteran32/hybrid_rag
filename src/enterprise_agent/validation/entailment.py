"""Local natural language inference on (source passage, proposed statement) pairs."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from onnxruntime import InferenceSession
    from tokenizers import Tokenizer

MODEL = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
REVISION = "8adb042d524ecd5c26d3e3ba0e3fbcf7e2d0864c"


@dataclass(frozen=True)
class Relation:
    entailment: float
    contradiction: float


Classifier = Callable[[list[tuple[str, str]]], list[Relation]]


class VerificationUnavailable(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _load() -> tuple[Tokenizer, InferenceSession, dict[str, int]]:
    import onnxruntime as ort
    from huggingface_hub import snapshot_download
    from tokenizers import Tokenizer

    graph = "onnx/model_quantized.onnx"
    root = Path(
        snapshot_download(
            MODEL,
            revision=REVISION,
            allow_patterns=["config.json", "tokenizer.json", graph],
        )
    )
    config = json.loads((root / "config.json").read_text())
    labels = {label: int(index) for index, label in config["id2label"].items()}
    tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
    tokenizer.enable_truncation(max_length=config["max_position_embeddings"])
    pad_id = config["pad_token_id"]
    tokenizer.enable_padding(pad_id=pad_id, pad_token=tokenizer.id_to_token(pad_id))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(
        str(root / graph), sess_options=options, providers=["CPUExecutionProvider"]
    )
    return tokenizer, session, labels


def classify_local(pairs: list[tuple[str, str]]) -> list[Relation]:
    import numpy as np

    try:
        tokenizer, session, labels = _load()
        names = [item.name for item in session.get_inputs()]
        fields = {
            "input_ids": "ids",
            "attention_mask": "attention_mask",
            "token_type_ids": "type_ids",
        }
        results = []
        for start in range(0, len(pairs), 8):
            batch = tokenizer.encode_batch(pairs[start : start + 8])
            inputs = {
                name: np.asarray(
                    [getattr(enc, fields[name]) for enc in batch], dtype=np.int64
                )
                for name in names
            }
            (logits,) = session.run(None, inputs)
            values = np.exp(logits - logits.max(axis=1, keepdims=True))
            probabilities = values / values.sum(axis=1, keepdims=True)
            results.extend(
                Relation(
                    float(row[labels["entailment"]]),
                    float(row[labels["contradiction"]]),
                )
                for row in probabilities
            )
        return results
    except Exception as exc:
        # The library boundary includes Hub errors and native ONNX exceptions.
        raise VerificationUnavailable(
            f"local evidence verifier unavailable ({type(exc).__name__}); install the answer extra and cache {MODEL}@{REVISION}"
        ) from exc
