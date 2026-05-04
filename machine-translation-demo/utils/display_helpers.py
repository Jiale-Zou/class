from __future__ import annotations

import html
from difflib import SequenceMatcher
from typing import List, Tuple


def word_diff_spans(a: str, b: str) -> Tuple[str, str]:
    a_words = [w for w in (a or "").split() if w != ""]
    b_words = [w for w in (b or "").split() if w != ""]
    sm = SequenceMatcher(a=a_words, b=b_words)

    a_out: List[str] = []
    b_out: List[str] = []

    def _wrap(words: List[str], bg: str) -> str:
        escaped = html.escape(" ".join(words))
        return f"<span style='background-color:{bg}; padding:0.1em 0.2em; border-radius:0.2em'>{escaped}</span>"

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        a_seg = a_words[i1:i2]
        b_seg = b_words[j1:j2]
        if tag == "equal":
            if a_seg:
                a_out.append(html.escape(" ".join(a_seg)))
            if b_seg:
                b_out.append(html.escape(" ".join(b_seg)))
        elif tag == "replace":
            if a_seg:
                a_out.append(_wrap(a_seg, "#ffe3e3"))
            if b_seg:
                b_out.append(_wrap(b_seg, "#e3f2ff"))
        elif tag == "delete":
            if a_seg:
                a_out.append(_wrap(a_seg, "#ffe3e3"))
        elif tag == "insert":
            if b_seg:
                b_out.append(_wrap(b_seg, "#e3f2ff"))

    a_html = " ".join([x for x in a_out if x.strip() != ""]).strip()
    b_html = " ".join([x for x in b_out if x.strip() != ""]).strip()
    return a_html, b_html


def block_html(text_html: str) -> str:
    return (
        "<div style='border:1px solid rgba(49,51,63,0.2); border-radius:0.5rem; padding:0.75rem; "
        "background:rgba(250,250,250,0.6); line-height:1.7'>"
        f"{text_html}"
        "</div>"
    )

