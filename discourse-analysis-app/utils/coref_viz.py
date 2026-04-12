from __future__ import annotations

import html
import random
from typing import Any, List, Sequence, Tuple

PALETTE = [
    "#b91c1c",
    "#1d4ed8",
    "#047857",
    "#a16207",
    "#7c3aed",
    "#0f766e",
    "#c2410c",
    "#be185d",
    "#0369a1",
    "#4d7c0f",
]


def assign_colors(n: int, seed: int = 42) -> List[str]:
    rng = random.Random(seed)
    cols = PALETTE * ((n // len(PALETTE)) + 1)
    rng.shuffle(cols)
    return cols[:n]


def clusters_to_rows(
    clusters: Sequence[Sequence[Tuple[int, int]]], text: str
) -> List[dict]:
    rows: List[dict] = []
    for cid, cl in enumerate(clusters, start=1):
        mentions = []
        for start, end in cl:
            span = text[start:end]
            mentions.append({"start": start, "end": end, "text": span})
        rows.append({"cluster_id": cid, "mentions": mentions, "size": len(mentions)})
    return rows


def render_coref_html(text: str, clusters: Sequence[Sequence[Tuple[int, int]]]) -> str:
    """
    Non-overlapping paint order: later clusters overlay earlier on overlaps
    (rare for coref). Mentions sorted by start desc for left-first paint.
    """
    colored = assign_colors(len(clusters))
    events: List[Tuple[int, int, int, str, bool]] = []
    for ci, cl in enumerate(clusters):
        color = colored[ci]
        ordered = sorted(cl, key=lambda se: (se[0], se[1]))
        for mi, (s, e) in enumerate(ordered):
            bold = mi == 0
            events.append((s, e, ci, color, bold))
    events.sort(key=lambda x: (x[0], -(x[1] - x[0])))

    # Build char array with nested spans is complex; use interval coloring:
    # For each char, track top cluster id by smallest span covering it.
    n = len(text)
    char_top: List[Tuple[int, str, bool]] = [(-1, "#111827", False) for _ in range(n)]
    for s, e, ci, color, bold in sorted(events, key=lambda x: (x[1] - x[0])):
        for p in range(s, min(e, n)):
            char_top[p] = (ci, color, bold)

    parts: List[str] = []
    p = 0
    while p < n:
        ci, color, bold = char_top[p]
        q = p
        while q < n and char_top[q] == (ci, color, bold):
            q += 1
        chunk = text[p:q]
        esc = html.escape(chunk)
        if ci >= 0:
            fw = "700" if bold else "500"
            title = f"Cluster {ci+1}"
            parts.append(
                f"<span title='{html.escape(title)}' style='background:{color}22;border-bottom:2px solid {color};"
                f"font-weight:{fw};color:#0f172a;padding:0 1px;border-radius:2px'>{esc}</span>"
            )
        else:
            parts.append(f"<span style='color:#0f172a'>{esc}</span>")
        p = q
    return f"<div style='font-size:16px;line-height:1.65'>{''.join(parts)}</div>"


def try_load_coref_nlp():
    """Returns (nlp, err_message)."""
    try:
        import spacy

        from fastcoref import spacy_component  # noqa: F401  # registers factory

        nlp = spacy.load(
            "en_core_web_sm",
            exclude=["parser", "lemmatizer", "ner", "textcat"],
        )
        if "parser" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
        nlp.add_pipe("fastcoref")
        return nlp, ""
    except Exception as exc:  # pragma: no cover - environment dependent
        msg = str(exc)
        if "all_tied_weights_keys" in msg:
            msg += (
                " — 多为 transformers 5.x 与 fastcoref 不兼容。请在当前 venv 执行: "
                "pip install \"transformers>=4.36,<5\""
            )
        return None, msg


def run_coref(nlp, text: str) -> Tuple[Any, List[List[Tuple[int, int]]]]:
    doc = nlp(text, component_cfg={"fastcoref": {"resolve_text": False}})
    clusters = doc._.coref_clusters or []
    return doc, clusters
