"""simai text -> ChartData (Phase C1).

Covers what charts in the collection use: (bpm) changes, {div} and {#sec}
commas, EACH ("/"), pseudo-EACH ("`"), two-digit taps ("15"), BREAK / EX /
star taps ($, $$), holds with [N:M], [#sec] and [bpm#N:M] lengths, touch
and touch holds with fireworks, slides with every shape, chains
("1-4-7[4:1]", "1-4[8:1]-7[8:1]"), several paths from one star ("*"),
grand V, wait overrides ([bpm#N:M], [bpm#sec], [sec##N:M], [sec##sec]),
star-less slides ("?", "!"), break slides, comments ("||") and the end "E".
Speed changes (<HS*x>) are skipped: they don't change when notes are hit.

Every time and length is kept exactly (no rounding). Anything else raises
ParseError naming the bad piece, so the collection check
(tools/check_codec.py) lists exactly what isn't understood.

repair=True also accepts four charting typos found in the collection, and
counts each fix in Issues: a tap or touch with a length ("3[4:1]" -> hold
"3h[4:1]", "Cf[1:1]" -> "Chf[1:1]"),
a doubled hold mark ("2hh[12:1]" -> "2h[12:1]") and an EX mark inside a
slide path ("7-4x[4:3]", "*x<4[8:1]" -> EX star).
"""
from __future__ import annotations

import re
from collections import Counter
from fractions import Fraction

from backend.chartcodec.notes import (ONE, ChartData, Hold, Path, Segment, Slide, Tap, Touch,
                                      TouchHold)


class ParseError(ValueError):
    pass


class Issues:
    """Things worth knowing about a chart: seconds-based values ("seconds",
    "other bpm"), dropped yellow-only 0s, typos fixed in repair mode."""

    def __init__(self, repair: bool = False):
        self.count: Counter = Counter()
        self.repair = repair

    def note(self, kind: str) -> None:
        self.count[kind] += 1

    def fix(self, kind: str, text: str) -> None:
        if not self.repair:
            raise ParseError(f"{kind} in {text!r}")
        self.count[f"fixed: {kind}"] += 1


def _num(text: str, what: str) -> Fraction:
    try:
        return Fraction(text.strip())
    except (ValueError, ZeroDivisionError):
        raise ParseError(f"bad number {text!r} in {what}") from None


def _ratio(text: str, what: str) -> Fraction:
    """'N:M' -> beats (M whole notes / N = 4M/N beats)."""
    n, _, m = text.partition(":")
    n, m = _num(n, what), _num(m, what)
    if n == 0 and m == 0:                                        # [0:0]: instant slide
        return Fraction(0)
    if n <= 0 or m < 0:
        raise ParseError(f"bad length {text!r} in {what}")
    return 4 * m / n


def duration(text: str, bpm: float, issues: Issues, what: str) -> tuple[Fraction | None, Fraction]:
    """Content of a [...] -> (wait in beats or None for the default, length in beats).
    Seconds are turned into beats with the current BPM."""
    sec = lambda s: _num(s, what) * Fraction(bpm) / 60          # seconds -> beats
    if "##" in text:
        w, _, l = text.partition("##")
        wait = sec(w)
        issues.note("seconds")
        length = _ratio(l, what) if ":" in l else sec(l)
    elif "#" in text:
        b, _, l = text.partition("#")
        if b.strip() == "":                                     # [#1.5]: seconds
            wait, length = None, sec(l)
            issues.note("seconds")
        else:
            k = Fraction(bpm) / _num(b, what)                   # beats at bpm b -> current beats
            wait = k
            issues.note("other bpm")
            length = _ratio(l, what) * k if ":" in l else sec(l)
    else:
        wait, length = None, _ratio(text, what)
    return wait, length


# ---------------------------------------------------------------- one note

_SHAPE = re.compile(r"pp|qq|[-^v<>pqszVw]")


def _flags(text: str, allowed: str, what: str) -> Counter:
    c = Counter(text)
    bad = set(c) - set(allowed)
    if bad:
        raise ParseError(f"unknown mark(s) {''.join(sorted(bad))!r} in {what!r}")
    return c


def _cw_of(symbol: str, start: int) -> str:
    """simai '>' turns clockwise from buttons 1, 2, 7, 8 and counter-clockwise
    from 3-6; '<' the opposite."""
    top = start in (1, 2, 7, 8)
    return "cw" if (symbol == ">") == top else "ccw"


def parse_note(text: str, beat: Fraction, bpm: float, sub: int, issues: Issues) -> list:
    t = text
    if t[0] in "ABCDE":
        return [_touch(t, beat, bpm, sub, issues)]
    if not t[0].isdigit() or t[0] in "09":
        raise ParseError(f"can't read note {text!r}")
    button = int(t[0])
    rest = t[1:]
    m = _SHAPE.search(re.sub(r"\[[^\]]*\]", lambda x: "#" * len(x.group()), rest))
    if m:
        return [_slide(button, rest[:m.start()], rest[m.start():], text, beat, bpm, sub, issues)]
    if "h" not in rest and "[" in rest:                          # "3[4:1]": a hold without its h
        issues.fix("tap with a length", text)
        a = rest.index("[")
        rest = rest[:a] + "h" + rest[a:]
    if "h" in rest:
        if rest.count("h") > 1:                                  # "2hh[12:1]"
            issues.fix("doubled h", text)
            i = rest.index("h")
            rest = rest[:i + 1] + rest[i + 1:].replace("h", "")
        head, _, tail = rest.partition("h")
        length = Fraction(0)
        if "[" in tail:
            a, j = tail.index("["), _close(tail, 0, "]")
            _, length = duration(tail[a + 1:j], bpm, issues, text)
            tail = tail[:a] + tail[j + 1:]
        f = _flags(head + tail, "bx", text)
        return [Hold(beat, button, length, brk=f["b"] > 0, ex=f["x"] > 0, sub=sub)]
    f = _flags(rest, "bx$", text)
    if f["$"] > 2:
        raise ParseError(f"too many $ in {text!r}")
    return [Tap(beat, button, brk=f["b"] > 0, ex=f["x"] > 0, star=f["$"], sub=sub)]


def _touch(t: str, beat, bpm, sub, issues: Issues) -> object:
    group = t[0]
    i = 1
    if i < len(t) and t[i].isdigit():
        idx = int(t[i]); i += 1
        if group == "C":
            if idx not in (1, 2):
                raise ParseError(f"bad sensor in {t!r}")
            sensor = "C"                                         # C1 / C2 are the same sensor
        else:
            if not 1 <= idx <= 8:
                raise ParseError(f"bad sensor in {t!r}")
            sensor = f"{group}{idx}"
    elif group == "C":
        sensor = "C"
    else:
        raise ParseError(f"sensor without a number in {t!r}")
    rest = t[i:]
    if "h" not in rest and "[" in rest:                          # "Cf[1:1]": a touch hold without its h
        issues.fix("touch with a length", t)
        a = rest.index("[")
        rest = rest[:a] + "h" + rest[a:]
    if "h" in rest:
        head, _, tail = rest.partition("h")
        length = Fraction(0)
        if "[" in tail:
            a, j = tail.index("["), _close(tail, 0, "]")
            _, length = duration(tail[a + 1:j], bpm, issues, t)
            tail = tail[:a] + tail[j + 1:]
        f = _flags(head + tail, "fx", t)
        return TouchHold(beat, sensor, length, fw=f["f"] > 0, ex=f["x"] > 0, sub=sub)
    f = _flags(rest, "fx", t)
    return Touch(beat, sensor, fw=f["f"] > 0, ex=f["x"] > 0, sub=sub)


def _slide(start, head_marks, body, text, beat, bpm, sub, issues) -> Slide:
    f = _flags(head_marks, "bx?!@$", text)
    heads = [h for h, mark in (("none", "?"), ("sudden", "!"), ("tap", "@"), ("pad", "$")) if f[mark]]
    if len(heads) > 1:
        raise ParseError(f"more than one of ? ! @ $ in {text!r}")
    head = heads[0] if heads else "star"
    paths, ex = [], f["x"] > 0
    for piece in body.split("*"):
        if not piece:
            raise ParseError(f"empty slide after * in {text!r}")
        bare = re.sub(r"\[[^\]]*\]", "", piece)
        if "x" in bare:                                          # "7-4x[4:3]", "*x<4[8:1]"
            issues.fix("EX mark inside a slide path", text)
            piece = re.sub(r"x(?![^\[]*\])", "", piece)
            ex = True
        paths.append(_path(start, piece, text, bpm, issues))
    return Slide(beat, start, tuple(paths), head=head, brk=f["b"] > 0, ex=ex, sub=sub)


def _path(start, s, text, bpm, issues) -> Path:
    segments, wait, brk, bracket = [], None, False, False
    prev, i = start, 0
    while i < len(s):
        c = s[i]
        if c == "b":                                             # break slide
            brk = True; i += 1; continue
        m = _SHAPE.match(s, i)
        if not m:
            raise ParseError(f"can't read slide {text!r} at {s[i:]!r}")
        sym = m.group(); i = m.end()
        mid = None
        if sym == "V":
            if i >= len(s) or not s[i].isdigit():
                raise ParseError(f"grand V without its turn in {text!r}")
            mid = int(s[i]); i += 1
        if i >= len(s) or s[i] not in "12345678":
            raise ParseError(f"slide shape without an end button in {text!r}")
        end = int(s[i]); i += 1
        while i < len(s) and s[i] == "b":
            brk = True; i += 1
        length = None
        if i < len(s) and s[i] == "[":
            j = _close(s, i, "]")
            w, length = duration(s[i + 1:j], bpm, issues, text)
            if w is not None:
                if bracket:
                    raise ParseError(f"wait given in a later [...] of {text!r}")
                wait = w
            bracket = True
            i = j + 1
        shape = _cw_of(sym, prev) if sym in "<>" else sym
        segments.append(Segment(shape, end, mid, length))
        prev = end
    if not segments:
        raise ParseError(f"slide without a path in {text!r}")
    if segments[-1].length is None:
        raise ParseError(f"slide without a length in {text!r}")
    return Path(tuple(segments), ONE if wait is None else wait, brk)


_OWNS_DIGIT = re.compile(r"(pp|qq|[-^v<>pqszVw]|V[1-8]|[ABDEC])$")


def split_notes(part: str) -> list[str]:
    """Notes written back to back without "/" are an EACH too: "15" is taps on
    1 and 5, "87-4[8:1]" a tap on 8 and a slide 7-4, "7<3[4:1]8p3[4:1]" two
    slides. A button digit starts a new note unless the note so far is waiting
    for it (after a slide shape, a grand V's turn, or a sensor letter)."""
    notes, cur, i = [], "", 0
    while i < len(part):
        c = part[i]
        if c == "[":
            j = part.find("]", i)
            if j < 0:
                raise ParseError(f"unclosed [ in {part!r}")
            cur += part[i:j + 1]; i = j + 1
            continue
        bare = re.sub(r"\[[^\]]*\]", "", cur)
        new = (c in "12345678" and bare and not _OWNS_DIGIT.search(bare)) or (c in "ABCDE" and bare)
        if new:
            notes.append(cur); cur = ""
        cur += c; i += 1
    if cur:
        notes.append(cur)
    return notes


# ---------------------------------------------------------------- whole chart

def _split_slot(raw: str) -> list[tuple[int, str]]:
    """'1/2`3' -> [(0,'1'), (0,'2'), (1,'3')], ignoring / and ` inside [...]."""
    out, cur, sub, depth = [], [], 0, 0
    for c in raw:
        if c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
        if depth == 0 and c in "/`":
            if cur:
                out.append((sub, "".join(cur)))
            cur = []
            if c == "`":
                sub += 1
            continue
        cur.append(c)
    if cur:
        out.append((sub, "".join(cur)))
    return out


def _close(text: str, i: int, ch: str) -> int:
    j = text.find(ch, i)
    if j < 0:
        raise ParseError(f"no closing {ch!r} for {text[i:i + 20]!r}")
    return j


def parse_chart(body: str, default_bpm: float | None = None,
                repair: bool = False) -> tuple[ChartData, Issues]:
    issues = Issues(repair)
    body = re.sub(r"\|\|[^\n]*", "", body)                       # comments
    chart = ChartData()
    beat, bpm, step = Fraction(0), default_bpm, Fraction(1)       # {4} by default
    sec_step = None
    if default_bpm:
        chart.tempo.append((Fraction(0), float(default_bpm)))
    token: list[str] = []

    def flush():
        raw = "".join(token).strip()
        token.clear()
        if not raw or raw == "E":
            return
        if bpm is None:
            raise ParseError(f"note {raw!r} before any (bpm)")
        at = beat
        for sub, part in _split_slot(raw):
            if part.strip() == "0":                              # "3/0": only makes 3 yellow
                issues.count["yellow-only 0 dropped"] += 1
                continue
            for piece in split_notes(part):
                chart.notes.extend(parse_note(piece, at, bpm, sub, issues))

    i, n = 0, len(body)
    while i < n:
        c = body[i]
        if c == "(":
            j = _close(body, i, ")")
            v = float(_num(body[i + 1:j], "bpm"))
            if v <= 0:
                raise ParseError(f"bad bpm ({body[i + 1:j]})")
            at = beat
            if not chart.tempo or chart.tempo[-1][1] != v:
                if chart.tempo and chart.tempo[-1][0] == at:
                    chart.tempo[-1] = (at, v)
                else:
                    chart.tempo.append((at, v))
            bpm = v
            i = j + 1
        elif c == "{":
            j = _close(body, i, "}")
            v = body[i + 1:j].strip()
            if v.startswith("#"):
                sec_step, step = _num(v[1:], "{#}"), None
            else:
                d = _num(v, "{}")
                if d <= 0:
                    raise ParseError(f"bad divisor {{{v}}}")
                step, sec_step = 4 / d, None
            i = j + 1
        elif c == "<" and body.startswith("<HS", i):                # speed change: skip
            i = _close(body, i, ">") + 1
        elif c == "[":
            j = _close(body, i, "]")
            token.append(body[i:j + 1]); i = j + 1
        elif c == ",":
            flush()
            if step is not None:
                beat += step
            else:
                if bpm is None:
                    raise ParseError("{#} comma before any (bpm)")
                beat += sec_step * Fraction(bpm) / 60
            i += 1
        elif c in " \t\r\n":
            i += 1
        else:
            token.append(c); i += 1
    flush()
    if not chart.tempo:
        raise ParseError("no (bpm) in the chart")
    return chart.canonical(), issues


def parse_maidata(text: str) -> dict:
    """maidata.txt -> {'first': float, 'title': str, 'wholebpm': float|None,
    'charts': {slot: {'level': str, 'body': str}}} (bodies not parsed yet)."""
    def field(name):
        m = re.search(rf"^&{name}=(.*)$", text, re.M)
        return m.group(1).strip() if m else ""
    first = field("first")
    wb = field("wholebpm")
    charts = {}
    for slot, body in re.findall(r"^&inote_(\d+)=(.*?)(?=^&|\Z)", text, re.M | re.S):
        charts[int(slot)] = {"level": field(f"lv_{slot}"), "body": body}
    try:
        whole = float(wb) if wb else None
    except ValueError:
        whole = None
    return {"first": float(first) if first else 0.0, "title": field("title"),
            "wholebpm": whole, "charts": charts}
