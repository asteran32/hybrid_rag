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
