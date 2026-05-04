from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple


def _simple_tokenize(text: str) -> List[str]:
    t = (text or "").strip()
    if not t:
        return []
    try:
        import nltk
        from nltk.tokenize import word_tokenize

        try:
            nltk.data.find("tokenizers/punkt")
        except Exception:
            try:
                nltk.download("punkt", quiet=True)
            except Exception:
                return re.findall(r"\w+|[^\w\s]", t, flags=re.UNICODE)
        return [x for x in word_tokenize(t) if x.strip()]
    except Exception:
        return re.findall(r"\w+|[^\w\s]", t, flags=re.UNICODE)


def _ngrams(tokens: Sequence[str], n: int) -> List[Tuple[str, ...]]:
    if n <= 0:
        return []
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(0, len(tokens) - n + 1)]


def _modified_precision(reference: Sequence[str], candidate: Sequence[str], n: int) -> Tuple[int, int, Counter]:
    cand_ngrams = Counter(_ngrams(candidate, n))
    ref_ngrams = Counter(_ngrams(reference, n))
    clipped: Counter = Counter()
    match = 0
    total = 0
    for ng, c_count in cand_ngrams.items():
        total += c_count
        clip = min(c_count, ref_ngrams.get(ng, 0))
        if clip > 0:
            clipped[ng] = clip
            match += clip
    return match, total, clipped


def _brevity_penalty(ref_len: int, cand_len: int) -> float:
    if cand_len <= 0:
        return 0.0
    if cand_len > ref_len:
        return 1.0
    return math.exp(1.0 - (ref_len / cand_len))


@dataclass(frozen=True)
class BleuDetails:
    bleu: float
    bp: float
    ref_len: int
    cand_len: int
    precisions: Dict[int, float]
    matches: Dict[int, int]
    totals: Dict[int, int]
    matched_ngrams_examples: Dict[int, List[str]]


def bleu_score_with_details(
    reference: str,
    candidate: str,
    max_n: int = 4,
    smooth: bool = True,
) -> BleuDetails:
    ref_toks = _simple_tokenize(reference)
    cand_toks = _simple_tokenize(candidate)

    bp = _brevity_penalty(len(ref_toks), len(cand_toks))
    precisions: Dict[int, float] = {}
    matches: Dict[int, int] = {}
    totals: Dict[int, int] = {}
    examples: Dict[int, List[str]] = {}

    logs: List[float] = []
    for n in range(1, max_n + 1):
        m, t, clipped = _modified_precision(ref_toks, cand_toks, n)
        matches[n] = int(m)
        totals[n] = int(t)
        if t == 0:
            p = 0.0
        else:
            if smooth:
                p = (m + 1.0) / (t + 1.0)
            else:
                p = m / t
        precisions[n] = float(p)
        if p <= 0.0:
            logs.append(float("-inf"))
        else:
            logs.append(math.log(p))
        ex = []
        for ng, c in clipped.most_common(10):
            ex.append(" ".join(ng) + f" ×{c}")
        examples[n] = ex

    if any(v == float("-inf") for v in logs):
        geo_mean = 0.0
    else:
        geo_mean = math.exp(sum(logs) / max_n) if max_n > 0 else 0.0

    bleu = bp * geo_mean
    return BleuDetails(
        bleu=float(bleu),
        bp=float(bp),
        ref_len=len(ref_toks),
        cand_len=len(cand_toks),
        precisions=precisions,
        matches=matches,
        totals=totals,
        matched_ngrams_examples=examples,
    )

