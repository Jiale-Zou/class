from __future__ import annotations

import html
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

CONNECTIVES: Dict[str, List[str]] = {
    "TEMPORAL": ["when", "after", "before", "while", "since", "until"],
    "CONTINGENCY": ["because", "since", "if", "unless", "so"],
    "COMPARISON": ["but", "although", "though", "however", "yet"],
    "EXPANSION": ["and", "or", "also", "furthermore", "moreover"],
}

CATEGORY_COLOR = {
    "TEMPORAL": "#1d4ed8",
    "CONTINGENCY": "#b91c1c",
    "COMPARISON": "#7c3aed",
    "EXPANSION": "#047857",
    "AMBIGUOUS": "#64748b",
}

MONTHS = {
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
}


def _window(tokens: List[str], i: int, left: int = 5, right: int = 5) -> Tuple[List[str], List[str]]:
    lo = max(0, i - left)
    hi = min(len(tokens), i + right + 1)
    return tokens[lo:i], tokens[i + 1 : hi]


def disambiguate_since(left_ctx: List[str], right_ctx: List[str]) -> str:
    joined_right = " ".join(right_ctx[:6])
    if re.match(r"^\d{4}\b", joined_right):
        return "TEMPORAL"
    if right_ctx and re.match(r"^\d{1,2}$", right_ctx[0]):
        return "TEMPORAL"
    if right_ctx and right_ctx[0].lower() in MONTHS:
        return "TEMPORAL"
    if right_ctx and right_ctx[0] in {
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
    }:
        return "TEMPORAL"
    if right_ctx and right_ctx[0] in {"it", "we", "they", "he", "she", "the", "there", "you", "i"}:
        return "CONTINGENCY"
    return "CONTINGENCY"


def disambiguate_so(right_ctx: List[str]) -> str:
    if right_ctx and right_ctx[0] in {"much", "many", "far", "little"}:
        return "EXPANSION"
    return "CONTINGENCY"


@dataclass
class ConnectiveHit:
    category: str
    token_index: int
    token_end_index: int
    lemma: str
    surface: str
    arg1: str
    arg2: str
    note: str = ""


def _phrase_index() -> Dict[str, set]:
    m: Dict[str, set] = defaultdict(set)
    for cat, words in CONNECTIVES.items():
        for w in words:
            m[w].add(cat)
    return m


def scan_connectives(doc) -> List[ConnectiveHit]:
    phrase_to_cats = _phrase_index()
    phrases = sorted(phrase_to_cats.keys(), key=lambda p: (-len(p.split()), len(p), p))
    tokens = list(doc)
    lowers = [t.text.lower() for t in tokens]
    hits: List[ConnectiveHit] = []
    i = 0
    while i < len(doc):
        if doc[i].is_space or doc[i].is_punct:
            i += 1
            continue
        matched = None
        for phrase in phrases:
            parts = phrase.split()
            n = len(parts)
            if i + n > len(lowers):
                continue
            if lowers[i : i + n] != parts:
                continue
            matched = (phrase, i, i + n - 1)
            break
        if not matched:
            i += 1
            continue
        phrase, start_i, end_i = matched
        cats = phrase_to_cats[phrase]
        left, right = _window(lowers, start_i, 5, 5)
        note = ""
        if phrase == "since" and cats & {"TEMPORAL", "CONTINGENCY"}:
            cat = disambiguate_since(left, right)
            note = "since 消歧"
        elif phrase == "so" and cats & {"CONTINGENCY", "EXPANSION"}:
            cat = disambiguate_so(right)
            note = "so 消歧"
        elif len(cats) == 1:
            cat = next(iter(cats))
        else:
            cat = sorted(cats)[0]
            note = "多标签：取字典序首个"
        sent = doc[start_i].sent
        arg1 = doc[sent.start : start_i].text.strip()
        arg2 = doc[end_i + 1 : sent.end].text.strip()
        surface = doc[start_i : end_i + 1].text
        hits.append(
            ConnectiveHit(
                category=cat,
                token_index=start_i,
                token_end_index=end_i,
                lemma=phrase,
                surface=surface,
                arg1=arg1,
                arg2=arg2,
                note=note,
            )
        )
        i = end_i + 1
    return hits


def render_connective_html(doc, hits: List[ConnectiveHit]) -> str:
    spans = {(h.token_index, h.token_end_index): h for h in hits}
    parts: List[str] = []
    i = 0
    while i < len(doc):
        hit = None
        for (s, e), h in spans.items():
            if i == s:
                hit = h
                break
        if hit:
            color = CATEGORY_COLOR.get(hit.category, "#111827")
            chunk = doc[hit.token_index : hit.token_end_index + 1].text
            label = html.escape(hit.category)
            parts.append(
                f"<span style='border-bottom:3px solid {color};font-weight:700;color:{color}' "
                f"title='{label}'>{html.escape(chunk)}</span>"
            )
            i = hit.token_end_index + 1
            continue
        parts.append(html.escape(doc[i].text_with_ws))
        i += 1
    return "".join(parts)


def render_args_blocks(hit: ConnectiveHit) -> str:
    a1 = html.escape(hit.arg1)
    a2 = html.escape(hit.arg2)
    mid = html.escape(hit.surface)
    return (
        "<div style='display:flex;flex-wrap:wrap;gap:8px;align-items:stretch'>"
        "<div style='flex:1;min-width:220px;background:#dcfce7;padding:10px;border-radius:6px'>"
        f"{a1}</div>"
        "<div style='background:#fef9c3;padding:10px;border-radius:6px;font-weight:700'>"
        f"{mid}</div>"
        "<div style='flex:1;min-width:220px;background:#dbeafe;padding:10px;border-radius:6px'>"
        f"{a2}</div>"
        "</div>"
    )
