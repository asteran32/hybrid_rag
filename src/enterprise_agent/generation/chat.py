import json
import os
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from pydantic import ValidationError

from enterprise_agent.generation.contracts import (
    AnswerDraft,
    Generation,
    GenerationInfo,
)
from enterprise_agent.generation.prompt import PROMPT_VERSION, SYSTEM_PROMPT

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletion

MAX_OUTPUT_TOKENS = 1200


class GenerationUnavailable(RuntimeError):
    """Configuration, transport, or provider failure. No answer was produced."""


def generate(question: str, passages: dict[str, str]) -> Generation:
    provider = os.getenv("ANSWER_PROVIDER") or "local"
    model = (os.getenv("ANSWER_MODEL") or "").strip()
    if provider == "local":
        base_url = os.getenv("ANSWER_BASE_URL") or "http://127.0.0.1:8080/v1"
        url = urlsplit(base_url)
        if url.scheme not in {"http", "https"} or url.hostname not in {
            "localhost",
            "127.0.0.1",
            "::1",
        }:
            raise GenerationUnavailable(
                "ANSWER_PROVIDER=local requires a loopback ANSWER_BASE_URL"
            )
        model = model or "local-model"
        api_key = "local"
    elif provider == "openai":
        base_url = "https://api.openai.com/v1"
        api_key = os.getenv("OPENAI_API_KEY")
        if not model or not api_key:
            raise GenerationUnavailable(
                "configure ANSWER_MODEL and OPENAI_API_KEY for OpenAI generation"
            )
    else:
        raise GenerationUnavailable("ANSWER_PROVIDER must be local or openai")
    # llama.cpp implements the legacy Chat Completions token-limit parameter.
    limits = (
        {"max_tokens": MAX_OUTPUT_TOKENS, "temperature": 0}
        if provider == "local"
        else {"max_completion_tokens": MAX_OUTPUT_TOKENS}
    )
    response = request_completion(
        model,
        answer_payload(question, passages),
        base_url=base_url,
        api_key=api_key,
        limits=limits,
        timeout=90.0,
    )
    return read_generation(response, provider, model)


def structured_payload(name: str, system: str, data: dict, schema: dict) -> dict:
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(data, ensure_ascii=False)},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": name, "strict": True, "schema": schema},
        },
    }


def answer_payload(question: str, passages: dict[str, str]) -> dict:
    schema = AnswerDraft.model_json_schema()
    schema["$defs"]["PassageRef"]["enum"] = list(passages)
    return structured_payload(
        "grounded_answer",
        SYSTEM_PROMPT,
        {"question": question, "passages": passages},
        schema,
    )


def request_completion(
    model: str,
    payload: dict,
    *,
    base_url: str,
    api_key: str,
    limits: dict,
    timeout: float,
) -> "ChatCompletion":
    try:
        from openai import APIError, OpenAI
    except ImportError as exc:
        raise GenerationUnavailable(
            "install the answer dependencies with uv sync --extra answer"
        ) from exc

    try:
        with OpenAI(
            base_url=base_url, api_key=api_key, max_retries=0, timeout=timeout
        ) as client:
            return client.chat.completions.create(model=model, **payload, **limits)
    except APIError as exc:
        status = getattr(exc, "status_code", None)
        raise GenerationUnavailable(
            f"answer provider failed ({type(exc).__name__}, status={status})"
        ) from None


def read_generation(
    response: "ChatCompletion", provider: str, model: str
) -> Generation:
    info = GenerationInfo(
        provider=provider,
        model=response.model or model,
        prompt_version=PROMPT_VERSION,
        input_tokens=response.usage.prompt_tokens if response.usage else None,
        output_tokens=response.usage.completion_tokens if response.usage else None,
    )
    if not response.choices:
        return Generation(outcome="invalid_model_output", info=info)
    choice = response.choices[0]
    if choice.message.refusal or choice.finish_reason == "content_filter":
        return Generation(outcome="model_refusal", info=info)
    if (
        choice.finish_reason == "length"
        or (info.output_tokens or 0) > MAX_OUTPUT_TOKENS
    ):
        return Generation(outcome="output_limit", info=info)
    if choice.finish_reason != "stop" or not choice.message.content:
        return Generation(outcome="invalid_model_output", info=info)
    try:
        draft = AnswerDraft.model_validate_json(choice.message.content)
    except ValidationError:
        return Generation(outcome="invalid_model_output", info=info)
    return Generation(outcome=draft, info=info)
