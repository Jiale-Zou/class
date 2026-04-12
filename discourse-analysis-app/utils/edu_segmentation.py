from __future__ import annotations

import html
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set, Tuple

import requests

NEURAL_EDU_RAW_BASE = (
    "https://raw.githubusercontent.com/PKU-TANGENT/NeuralEDUSeg/master/data/rst/TRAINING"
)

# Curated small samples (document .out + aligned .out.edus ground truth)
SAMPLE_FILES = [
    "wsj_0601.out",
    "wsj_0602.out",
    "wsj_0603.out",
    "wsj_0604.out",
    "wsj_0605.out",
]


def cache_dir() -> Path:
    p = Path(__file__).resolve().parent.parent / "data" / "cache"
    p.mkdir(parents=True, exist_ok=True)
    return p


def fetch_or_load_cached(url: str, path: Path, timeout: int = 60) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8", errors="replace")
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    text = r.text
    path.write_text(text, encoding="utf-8")
    return text


def fetch_sample_pair(stem: str) -> Tuple[str, str]:
    """
    Return (plain_out, edus) for a stem like 'wsj_0601.out'.
    Plain comes from .out; ground-truth lines from .out.edus.
    """
    assert stem.endswith(".out")
    base = stem[:-4]  # wsj_0601
    d = cache_dir()
    out_url = f"{NEURAL_EDU_RAW_BASE}/{base}.out"
    edus_url = f"{NEURAL_EDU_RAW_BASE}/{base}.out.edus"
    out_text = fetch_or_load_cached(out_url, d / f"{base}.out")
    edus_text = fetch_or_load_cached(edus_url, d / f"{base}.out.edus")
    return out_text, edus_text


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def parse_edu_xml(text: str) -> Optional[List[str]]:
    """If <EDU> tags exist, return segments; else None."""
    if "<EDU" not in text.upper():
        return None
    pattern = re.compile(r"<EDU[^>]*>(.*?)</EDU>", re.IGNORECASE | re.DOTALL)
    parts = [normalize_ws(m.group(1)) for m in pattern.finditer(text)]
    parts = [p for p in parts if p]
    return parts or None


def parse_edu_lines(edus_file: str) -> List[str]:
    """NeuralEDUSeg .out.edus: one EDU per line."""
    segs: List[str] = []
    for line in edus_file.splitlines():
        s = line.strip()
        if s:
            segs.append(s)
    return segs


def canonical_from_edus(segments: Sequence[str]) -> str:
    return normalize_ws(" ".join(segments))


def char_starts_for_segments(segments: Sequence[str]) -> List[int]:
    """Start char offset of each segment in canonical join with single spaces."""
    starts: List[int] = []
    off = 0
    for i, seg in enumerate(segments):
        starts.append(off)
        off += len(seg)
        if i < len(segments) - 1:
            off += 1
    return starts


def char_to_token_before(doc, char_pos: int) -> int:
    """Index of first token whose span starts at or after char_pos."""
    for tok in doc:
        if tok.idx >= char_pos:
            return tok.i
    return len(doc)


def gold_boundaries_before_tokens(doc, segment_starts_chars: Sequence[int]) -> Set[int]:
    """
    Boundaries as token indices `j` meaning EDU starts at token j.
    Always includes 0 implicitly; we return the set of j>0.
    """
    b: Set[int] = set()
    for c in segment_starts_chars[1:]:
        b.add(char_to_token_before(doc, c))
    return b


@dataclass
class RuleFlags:
    punct: bool = True
    mark_cc: bool = True
    sconj: bool = True
    root: bool = False


def predict_boundaries(doc, flags: RuleFlags) -> Set[int]:
    """
    Predict EDU starts as token indices (same convention as gold_boundaries_before_tokens).
    """
    boundaries: Set[int] = set()
    n = len(doc)
    for tok in doc:
        if flags.punct and tok.pos_ == "PUNCT" and tok.text in {".", "?", "!"}:
            nxt = tok.i + 1
            if nxt < n:
                boundaries.add(nxt)
            continue
        if tok.i == 0:
            continue
        if flags.mark_cc and tok.dep_ in {"mark", "cc"}:
            boundaries.add(tok.i)
            continue
        if flags.sconj and tok.pos_ == "SCONJ":
            boundaries.add(tok.i)
            continue
        if flags.root and tok.dep_ == "ROOT" and tok.head.i == tok.i and tok.i > tok.sent.start:
            boundaries.add(tok.i)
    boundaries.discard(0)
    return boundaries


def explain_boundary(doc, j: int, flags: RuleFlags) -> str:
    """Heuristic label for a predicted boundary immediately before token j."""
    if j <= 0 or j >= len(doc):
        return "other"
    prev = doc[j - 1]
    if flags.punct and prev.pos_ == "PUNCT" and prev.text in {".", "?", "!"}:
        return "punct"
    tok = doc[j]
    if flags.mark_cc and tok.dep_ in {"mark", "cc"}:
        return "mark_cc"
    if flags.sconj and tok.pos_ == "SCONJ":
        return "sconj"
    if flags.root and tok.dep_ == "ROOT" and tok.head.i == tok.i and tok.i > tok.sent.start:
        return "root"
    return "other"


def boundaries_to_segments(doc, starts: Set[int]) -> List[str]:
    """Token-index starts (including 0) -> segment strings."""
    ordered = sorted({0} | set(starts))
    ordered = [i for i in ordered if 0 <= i < len(doc)]
    if not ordered:
        return [doc.text]
    pieces: List[str] = []
    for a, b in zip(ordered, ordered[1:] + [len(doc)]):
        span = doc[a:b]
        pieces.append(span.text.strip())
    return [p for p in pieces if p]


def boundary_f1(pred: Set[int], gold: Set[int], n_tokens: int) -> Tuple[float, float, float]:
    """
    Micro-F1 over *between-token* boundary indicators for positions 1..n_tokens-1.
    A position j in pred/gold means a boundary immediately before token j.
    """
    inner = set(range(1, max(1, n_tokens)))
    pset = pred & inner
    gset = gold & inner
    tp = len(pset & gset)
    fp = len(pset - gset)
    fn = len(gset - pset)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return prec, rec, f1


def length_bucket(n_chars: int) -> str:
    if n_chars < 40:
        return "<40"
    if n_chars < 90:
        return "40-90"
    if n_chars < 160:
        return "90-160"
    return ">=160"


def per_segment_length_analysis(
    gold_segments: Sequence[str], pred_segments: Sequence[str]
) -> List[dict]:
    """Rough alignment by index for length demo (not semantic alignment)."""
    rows: List[dict] = []
    m = min(len(gold_segments), len(pred_segments))
    for i in range(m):
        g = gold_segments[i]
        p = pred_segments[i]
        rows.append(
            {
                "segment_index": i,
                "gold_len": len(g),
                "pred_len": len(p),
                "gold_bucket": length_bucket(len(g)),
                "match_len": int(len(g) == len(p)),
            }
        )
    return rows


def highlight_boundary_words(
    doc,
    boundary_tokens: Iterable[int],
    *,
    highlight_boundaries: bool,
) -> str:
    """Render doc as HTML with boundary tokens highlighted."""
    bset = set(boundary_tokens)
    parts: List[str] = []
    for tok in doc:
        t = html.escape(tok.text)
        if highlight_boundaries and tok.i in bset:
            parts.append(
                f"<span style='background:#fde047;font-weight:700;padding:0 2px;border-radius:2px'>{t}</span>"
            )
        else:
            parts.append(t)
        if tok.whitespace_:
            parts.append(html.escape(tok.whitespace_))
    return "".join(parts)


def edu_cards_html(
    segments: Sequence[str],
    *,
    border_color: str,
    title: str,
) -> str:
    cards: List[str] = []
    for i, seg in enumerate(segments):
        body = html.escape(seg)
        cards.append(
            f"<div style='margin:8px 0;padding:10px 12px;border-left:4px solid {border_color};"
            f"background:#f8fafc;border-radius:6px;box-shadow:0 1px 2px rgba(0,0,0,.06)'>"
            f"<div style='font-size:12px;color:#64748b;margin-bottom:4px'>{html.escape(title)} #{i+1}</div>"
            f"<div style='font-size:15px;line-height:1.45;color:#0f172a'>{body}</div></div>"
        )
    return "".join(cards)
