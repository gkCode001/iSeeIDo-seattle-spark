"""``Task.active`` — the one deliberate exception to UTC-everywhere.

SPEC §6.1 gives a task a **local** wall-clock window, ``"18:00-06:00"``, and it may wrap
midnight. A human writing "overnight" means their night, not UTC's, so the window is
stored exactly as typed and resolved against ``ui.display_timezone`` **at match time**.

Storing a UTC-converted copy is the obvious optimisation and it is wrong: the offset a
window converts to is only valid until the next DST transition, after which a task set to
watch 18:00–06:00 quietly starts watching 17:00–05:00 and nobody finds out until the
alert that should have fired did not. The conversion is microseconds; the bug is a
season long.

Direction matters. ``shared.timecode.to_local`` maps an absolute instant to exactly one
local time and cannot throw on a DST boundary — that is the safe direction and the one
used here. The dangerous direction, local-naive → UTC, is ambiguous inside a fall-back
hour and duplicated across a spring-forward one; it deliberately does not exist in this
module, because there is no correct answer to "which 01:30 did you mean".

Comparison is against the chunk's ``t_start`` rendered in the display zone. Half-open at
both ends in the usual way: a window is entered at its start second and left at its end
second, so ``18:00-06:00`` contains 18:00:00 and does not contain 06:00:00.
"""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

from shared.timecode import to_local

__all__ = ["ActiveWindow", "ActiveWindowError", "parse_active_window", "ALWAYS"]

#: ``HH:MM-HH:MM``. Hours 00–24 so that a full day can be written the way SPEC §6.1's
#: default writes it; minutes 00–59. Whitespace around the dash is tolerated because a
#: human typed this into a form (SPEC §11.3).
_SPEC_RE = re.compile(r"^\s*(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})\s*$")

_MINUTES_PER_DAY = 24 * 60


class ActiveWindowError(ValueError):
    """``active`` is not a parseable local wall-clock window."""


@dataclass(frozen=True)
class ActiveWindow:
    """A local wall-clock window, possibly wrapping midnight.

    ``spec`` is kept verbatim so the UI can render what the human wrote (SPEC §11.3
    renders ``active`` literally and never converts it).
    """

    spec: str
    start_minute: int
    end_minute: int

    @property
    def wraps_midnight(self) -> bool:
        """True for ``18:00-06:00``: the window is the *complement* of 06:00–18:00."""
        return self.end_minute < self.start_minute

    @property
    def always(self) -> bool:
        """True for ``00:00-24:00`` and for any window whose ends coincide.

        Ends coinciding is read as "all day" rather than "no time at all". A task nobody
        can ever trigger is a silent failure; a task that is always armed is visible in
        the Watch pane within one poll.
        """
        return self.start_minute == self.end_minute % _MINUTES_PER_DAY and (
            self.end_minute - self.start_minute
        ) % _MINUTES_PER_DAY == 0

    def contains_local_minute(self, minute_of_day: float) -> bool:
        """Pure arithmetic form, for tests and for reasoning about the wrap."""
        if self.always:
            return True
        if self.wraps_midnight:
            return minute_of_day >= self.start_minute or minute_of_day < self.end_minute
        return self.start_minute <= minute_of_day < self.end_minute

    def contains(self, instant: datetime, tz: str | ZoneInfo | None = None) -> bool:
        """Is this absolute UTC instant inside the window, in the display timezone?

        ``tz`` overrides ``ui.display_timezone`` — for tests, and for the day this box
        watches a camera in a different city than the one it sits in.
        """
        local = to_local(instant, tz)
        return self.contains_local_minute(local.hour * 60 + local.minute + local.second / 60.0)

    def to_dict(self) -> dict[str, object]:
        return {
            "active": self.spec,
            "start_minute": self.start_minute,
            "end_minute": self.end_minute,
            "wraps_midnight": self.wraps_midnight,
        }


@functools.lru_cache(maxsize=64)
def parse_active_window(spec: str) -> ActiveWindow:
    """Parse ``"18:00-06:00"``. Cached — this runs per task per chunk, forever.

    Rejects rather than guesses. ``"overnight"``, ``"18:00–06:00"`` with an en dash, and
    ``"6pm-6am"`` are all things a human will type into the SPEC §11.3 form, and silently
    reading any of them as "always on" would turn a scoped task into an unscoped one
    without saying so.
    """
    match = _SPEC_RE.match(spec or "")
    if match is None:
        raise ActiveWindowError(
            f"active window must be local HH:MM-HH:MM (it may wrap midnight, e.g. "
            f"'18:00-06:00'); got {spec!r}"
        )
    sh, sm, eh, em = (int(g) for g in match.groups())
    for hour, minute in ((sh, sm), (eh, em)):
        if hour > 24 or minute > 59 or (hour == 24 and minute != 0):
            raise ActiveWindowError(f"active window has an impossible time: {spec!r}")
    return ActiveWindow(
        spec=spec.strip(),
        # 24:00 is a legal way to write "end of day" and is the SPEC §6.1 default's upper
        # bound. Normalising it to 0 here would turn "00:00-24:00" into a zero-length
        # window, so it is kept as 1440 and `always` reads the span, not the endpoints.
        start_minute=sh * 60 + sm,
        end_minute=eh * 60 + em,
    )


#: The SPEC §6.1 default — a task with no stated hours watches all of them.
ALWAYS = parse_active_window("00:00-24:00")
