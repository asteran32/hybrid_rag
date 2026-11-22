import json

import pytest

from enterprise_agent.retrieval.pageindex import source_passages
from enterprise_agent.retrieval.pageindex_selector import (
    LocalSelector,
    Navigation,
    Selection,
    SelectorSettings,
)


class Script:
    def __init__(self, replies):
        self.replies = iter(replies)

    def select(self, question, stage, choices, limit, timeout):
        return Selection(
            next(self.replies), {"prompt_tokens": 2, "completion_tokens": 1}
        )


def test_unknown_duplicate_and_boolean_ids_do_not_escape_batch():
    nav = Navigation(Script([[True], [2]]), SelectorSettings(max_calls=2), "q")
    options = [("left", "a"), ("right", "b")]
    assert nav.choose("document", options, 2) == ["right"]
    assert (nav.trace.retries, nav.trace.calls, nav.trace.input_tokens) == (1, 2, 4)
    assert nav.choose("node", options, 2) == []
    assert nav.trace.stop_reason == "budget_exhausted"
    nav = Navigation(Script([[1, 1], [3]]), SelectorSettings(), "q")
    assert nav.choose("document", options, 2) == []
    assert nav.trace.stop_reason == "invalid_selection"
    with pytest.raises(ValueError, match="loopback"):
        LocalSelector(SelectorSettings(base_url="https://example.com/v1"))


def test_different_lines_and_long_sentences_keep_original_offsets():
    text = "  🌲 one.\r\n" + "The item. " * 50
    ranges = source_passages(text, 100)
    assert text[ranges[0][0] - 100 : ranges[0][1] - 100] == "🌲 one."
    assert len(ranges) == 51
    assert all(text[a - 100 : b - 100] == "The item." for a, b in ranges[1:])


def test_tokenizer_preserves_budget_for_repeated_multibyte_terms(tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers

    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "근거": 1}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.enable_truncation(max_length=2)
    path = tmp_path / "tokenizer.json"
    tokenizer.save(str(path))
    settings = SelectorSettings(tokenizer_path=path, context_tokens=1024)
    nav = Navigation(Script([[1]]), settings, "q")
    label = "근거 " * 120
    assert nav._units(label) == 120
    assert nav.choose("evidence", [("real", label)], 1) == ["real"]
    fallback = Navigation(
        Script([]), settings.model_copy(update={"tokenizer_path": None}), "q"
    )
    assert fallback.choose("evidence", [("real", label)], 1) == []
    assert fallback.trace.stop_reason == "context_exhausted"


def test_local_selector_accepts_only_labels_in_this_call(monkeypatch):
    from enterprise_agent.retrieval import pageindex_selector as module

    returns_unknown = False

    class Endpoint:
        status = 200

        def __init__(self, *args, **kwargs):
            pass

        def request(self, method, path, body, headers):
            data = json.loads(body)
            self.allowed = data["response_format"]["json_schema"]["schema"][
                "properties"
            ]["ids"]["items"]["enum"]
            assert len(set(self.allowed)) == 2
            assert method == "POST" and path == "/v1/chat/completions"

        def getresponse(self):
            return self

        def read(self, limit):
            key = "unlisted option" if returns_unknown else self.allowed[1]
            return json.dumps(
                {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"content": json.dumps({"ids": [key]})},
                        }
                    ]
                }
            ).encode()

        def close(self):
            pass

    monkeypatch.setattr(module.http.client, "HTTPConnection", Endpoint)
    selector = LocalSelector(SelectorSettings())
    choices = ["Repeated title | one", "Repeated title | two"]
    assert selector.select("q", "document", choices, 2, 1).ids == [2]
    returns_unknown = True
    assert selector.select("q", "document", choices, 2, 1).ids is None
