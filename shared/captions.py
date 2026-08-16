"""Splitting a caption into its description and its watchlist verdicts.

A task-aware caption (see ``services/ingest/watchlist.py``) has two parts that must never
be treated as one body of text::

    Four individuals are seated around a table... A doorway is visible on the right.

    WATCHING:
    - a vehicle stopped in front of the fire door: absent
    - a person unloading boxes from a vehicle at the loading bay: absent
    - a person wearing a red wristband: present

The description is prose about the footage. The WATCHING block is a set of answers, and it
**quotes every task's wording back verbatim — including the tasks that are absent**.

That distinction is not cosmetic. Measured on the caption above, against
``config/tasks.yaml`` at the 0.15 gate:

======================= ============== ===========
task                    full caption   description
======================= ============== ===========
fire-door-blocked       0.38 HIT       0.19 HIT
loading-bay-activity    0.40 HIT       0.15 HIT
red-tag-person          0.30 HIT       0.11
======================= ============== ===========

Only ``red-tag-person`` is actually present. Embedding the whole caption does not merely
blur stage 1, it **inverts** it: the two absent tasks outscore the present one, because
their descriptions are longer and echo more distinctive words. Cosine cannot see the word
"absent" three tokens away; it sees "vehicle", "fire door", "loading bay" and matches.

So the two parts get used differently, and this module is the only place that splits them:

* **M5 stage 1** reads the verdict. ``present`` is a hit, ``absent`` is a miss — an exact
  answer from the model that saw the pixels, replacing a word-overlap heuristic. This is
  strictly better than the cosine it supersedes, which is why the fallback below matters
  only for captions written before the watchlist existed.
* **M2** embeds the description only. Every caption ending in the same three task lines
  would otherwise share a constant chunk of vector, which flattens the differences search
  exists to find and drags unrelated footage into every result.

Captions with no WATCHING block parse to ``watching={}`` and the description is the whole
text, so nothing here changes how pre-watchlist captions behave.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

__all__ = ["CaptionParts", "split_caption", "PRESENT", "ABSENT"]

PRESENT = "present"
ABSENT = "absent"

#: The header the caption prompt asks for. Tolerant of case, leading bullets and bold
#: markers, because a VLM writes "WATCHING:", "**WATCHING:**" and "- Watching:" on
#: different days and losing the whole block to a stray asterisk would silently disable
#: every standing task.
_HEADER_RE = re.compile(r"^[\s\-\*#>]*\**\s*watching\s*\**\s*:", re.IGNORECASE | re.MULTILINE)

#: One verdict line: an item, a separator, an answer. The answer is matched at the END so
#: that a description containing the word "present" does not decide the verdict.
_LINE_RE = re.compile(
    r"^[\s\-\*•\d\.\)]*(?P<item>.+?)\s*[:—\-]\s*\**(?P<verdict>present|absent|yes|no|not\s+present|not\s+visible)\**\s*[\.\,]?\s*$",
    re.IGNORECASE,
)

_TRUE = {"present", "yes"}


def _normalise(text: str) -> set[str]:
    """Token set for loose item matching, minus the words every task line shares."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in _STOP}


_STOP = {"a", "an", "the", "in", "of", "at", "on", "from", "is", "are", "and", "or", "to"}


@dataclass(frozen=True)
class CaptionParts:
    """A caption's prose and its per-item verdicts."""

    description: str
    watching: dict[str, str] = field(default_factory=dict)

    @property
    def has_watchlist(self) -> bool:
        """False for a caption written before the watchlist, or when the model ignored it.

        Callers use this to decide whether to trust :meth:`verdict_for` or fall back to
        the cosine path — an empty ``watching`` means "no answer", never "absent".
        """
        return bool(self.watching)

    def verdict_for(self, describe: str) -> bool | None:
        """Did the model say this task's item is present?

        ``True``/``False`` when the item was answered, ``None`` when it was not — and the
        distinction matters: ``None`` must fall back to cosine rather than be read as
        "absent", or a model that skipped a line would silently disarm a standing task.

        Matched on token overlap rather than string equality. The prompt quotes the task
        description verbatim, but models re-wrap, re-punctuate and drop articles, and a
        task that stops matching its own checklist line fails silently.
        """
        want = _normalise(describe)
        if not want:
            return None
        best: tuple[float, str] | None = None
        for item, verdict in self.watching.items():
            got = _normalise(item)
            if not got:
                continue
            overlap = len(want & got) / len(want)
            if best is None or overlap > best[0]:
                best = (overlap, verdict)
        # 0.6 admits a dropped article or a re-worded tail; it rejects two different tasks
        # that merely share a noun ("a person ..." vs "a person ...").
        if best is None or best[0] < 0.6:
            return None
        return best[1] in _TRUE


def split_caption(text: str | None) -> CaptionParts:
    """Split a caption into prose and verdicts. Never raises; unparseable input is prose."""
    if not text:
        return CaptionParts(description="", watching={})

    match = _HEADER_RE.search(text)
    if match is None:
        return CaptionParts(description=text.strip(), watching={})

    description = text[: match.start()].strip()
    watching: dict[str, str] = {}
    for raw in text[match.end() :].splitlines():
        line = raw.strip()
        if not line:
            continue
        hit = _LINE_RE.match(line)
        if hit is None:
            continue
        item = hit.group("item").strip().strip("*").strip()
        verdict = hit.group("verdict").strip().lower()
        if not item:
            continue
        watching[item] = PRESENT if verdict in _TRUE else ABSENT

    # A header with nothing parseable under it is a malformed caption, not an empty
    # watchlist. Keeping the raw text as the description means the block still reaches
    # search and stage 2 rather than being dropped on the floor.
    if not watching:
        return CaptionParts(description=text.strip(), watching={})
    return CaptionParts(description=description or text.strip(), watching=watching)
