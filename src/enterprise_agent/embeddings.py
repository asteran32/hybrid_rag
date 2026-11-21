import math
import os
import re
from typing import Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

DEFAULT_MODEL = "minishlab/potion-base-8M"
DEFAULT_REVISION = "bf8b056651a2c21b8d2565580b8569da283cab23"


class ModelSpec(BaseModel):
    model_config = ConfigDict(frozen=True)
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    dimensions: int = Field(gt=0, le=16000)
    revision: str | None = None

    @property
    def key(self) -> str:
        return self.model_dump_json()

    @model_validator(mode="after")
    def check_configuration(self) -> Self:
        if self.provider == "model2vec" and not re.fullmatch(
            "[a-f0-9]{40}", self.revision or ""
        ):
            raise ValueError(
                "configure the local model's EMBEDDING_REVISION (Hub commit hash)"
            )
        if self.provider == "openai":
            if self.revision:
                raise ValueError("unset EMBEDDING_REVISION for OpenAI")
            maximum = {
                "text-embedding-3-small": 1536,
                "text-embedding-3-large": 3072,
            }.get(self.model)
            if maximum and self.dimensions > maximum:
                raise ValueError(f"{self.model} supports at most {maximum} dimensions")
            if self.model == "text-embedding-ada-002" and self.dimensions != 1536:
                raise ValueError("text-embedding-ada-002 requires 1536 dimensions")
        return self


class Embedder(Protocol):
    spec: ModelSpec

    def embed(self, texts: list[str]) -> list[list[float]]: ...


def checked_vectors(provider: Embedder, texts: list[str]) -> list[list[float]]:
    vectors = provider.embed(texts)
    if len(vectors) != len(texts):
        raise ValueError("embedding provider returned the wrong number of vectors")
    for vector in vectors:
        if len(vector) != provider.spec.dimensions:
            raise ValueError(
                f"expected {provider.spec.dimensions} embedding dimensions, got {len(vector)}"
            )
        if not all(math.isfinite(x) for x in vector) or not any(vector):
            raise ValueError("cosine search requires finite nonzero embeddings")
    return vectors


class LocalEmbeddings:
    def __init__(self, spec: ModelSpec):
        from huggingface_hub import snapshot_download
        from model2vec import StaticModel

        self.spec = spec
        path = snapshot_download(
            spec.model,
            revision=spec.revision,
            allow_patterns=["*.json", "model.safetensors"],
        )
        self.model = StaticModel.from_pretrained(path)
        if self.model.dim != spec.dimensions:
            raise ValueError(
                f"model has {self.model.dim} dimensions; configured {spec.dimensions}"
            )

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, use_multiprocessing=False).tolist()


class OpenAIEmbeddings:
    def __init__(self, spec: ModelSpec):
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("set OPENAI_API_KEY to use OpenAI embeddings")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ValueError("install the adapter with uv sync --extra openai") from exc
        self.spec = spec
        self.client = OpenAI(max_retries=0, timeout=60.0)

    def embed(self, texts: list[str]) -> list[list[float]]:
        from openai import APIError

        kwargs = (
            {}
            if self.spec.model == "text-embedding-ada-002"
            else {"dimensions": self.spec.dimensions}
        )
        try:
            response = self.client.embeddings.create(
                model=self.spec.model, input=texts, **kwargs
            )
        except APIError as exc:
            status = getattr(exc, "status_code", None)
            raise ValueError(
                f"OpenAI embedding request failed ({type(exc).__name__}, status={status})"
            ) from None
        ordered = sorted(response.data, key=lambda item: item.index)
        if [item.index for item in ordered] != list(range(len(texts))):
            raise ValueError("OpenAI returned inconsistent embedding indices")
        return [item.embedding for item in ordered]


def configured_embedder() -> Embedder:
    name = os.getenv("EMBEDDING_PROVIDER", "model2vec")
    if name not in {"openai", "model2vec"}:
        raise ValueError(f"unknown embedding provider: {name}")
    local = name == "model2vec"
    model = os.getenv("EMBEDDING_MODEL") or (
        DEFAULT_MODEL if local else "text-embedding-3-small"
    )
    revision = os.getenv("EMBEDDING_REVISION") or (
        DEFAULT_REVISION if local and model == DEFAULT_MODEL else None
    )
    spec = ModelSpec(
        provider=name,
        model=model,
        revision=revision,
        dimensions=int(os.getenv("EMBEDDING_DIMENSIONS") or (256 if local else 1536)),
    )
    return LocalEmbeddings(spec) if local else OpenAIEmbeddings(spec)
