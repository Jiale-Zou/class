import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import nltk


def _ensure_nltk() -> None:
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)

    try:
        nltk.data.find("corpora/reuters")
    except LookupError:
        nltk.download("reuters", quiet=True)


def load_nltk_sample_corpus(max_chars: int = 20000) -> str:
    _ensure_nltk()
    from nltk.corpus import reuters

    file_ids = reuters.fileids()[:40]
    texts = [reuters.raw(fid) for fid in file_ids]
    text = "\n".join(texts)
    return text[:max_chars]


def corpus_stats(text: str) -> Dict[str, int]:
    _ensure_nltk()
    chars = len(text)
    words = 0
    sentences = 0
    vocab = 0
    if text.strip():
        sents = nltk.sent_tokenize(text)
        sentences = len(sents)
        tokens = nltk.word_tokenize(text)
        words = len(tokens)
        vocab = len(set([t.lower() for t in tokens if re.search(r"\w", t)]))
    return {"chars": chars, "words": words, "sentences": sentences, "vocab": vocab}


def _tokenize_words(text: str) -> List[str]:
    _ensure_nltk()
    tokens = nltk.word_tokenize(text)
    tokens = [t.lower() for t in tokens if re.search(r"\w", t)]
    return tokens


def prepare_corpus_text(text: str, unk_mode: str = "不处理") -> str:
    tokens = _tokenize_words(text)
    if not tokens:
        return ""

    if unk_mode == "不处理":
        return " ".join(tokens)

    threshold = 2 if "freq<2" in unk_mode else 3
    counts = Counter(tokens)
    normalized = [t if counts[t] >= threshold else "<unk>" for t in tokens]
    return " ".join(normalized)


@dataclass(frozen=True)
class NgramModel:
    n: int
    vocab: List[str]
    vocab_size: int
    ngram_counts: Counter
    context_counts: Counter


def build_ngram_model(corpus_text: str, n: int = 3) -> NgramModel:
    tokens = _tokenize_words(corpus_text)
    if not tokens:
        return NgramModel(
            n=n,
            vocab=[],
            vocab_size=0,
            ngram_counts=Counter(),
            context_counts=Counter(),
        )

    ngram_counts: Counter = Counter()
    context_counts: Counter = Counter()

    start = ["<s>"] * (n - 1)
    end = ["</s>"]

    padded = start + tokens + end
    for i in range(len(padded) - n + 1):
        ng = tuple(padded[i : i + n])
        ctx = ng[:-1]
        ngram_counts[ng] += 1
        context_counts[ctx] += 1

    vocab = sorted(set(tokens + ["<unk>"]))
    return NgramModel(
        n=n,
        vocab=vocab,
        vocab_size=len(vocab),
        ngram_counts=ngram_counts,
        context_counts=context_counts,
    )


def _sentence_tokens(sentence: str) -> List[str]:
    tokens = _tokenize_words(sentence)
    return tokens


def _ngram_sequence(tokens: Sequence[str], n: int) -> List[Tuple[str, ...]]:
    start = ["<s>"] * (n - 1)
    end = ["</s>"]
    padded = start + list(tokens) + end
    return [tuple(padded[i : i + n]) for i in range(len(padded) - n + 1)]


def sentence_logprob_details(
    sentence: str, model: NgramModel, use_laplace: bool
) -> Dict[str, Any]:
    tokens = _sentence_tokens(sentence)
    if not tokens or model.vocab_size == 0:
        return {
            "n": model.n,
            "logprob": -1e9,
            "had_zero": True,
            "rows": [],
        }

    rows: List[Dict[str, Any]] = []
    logprob = 0.0
    had_zero = False

    for ng in _ngram_sequence(tokens, model.n):
        ctx = ng[:-1]
        target = ng[-1]
        ctx_count = model.context_counts.get(ctx, 0)
        ng_count = model.ngram_counts.get(ng, 0)

        if use_laplace:
            denom = ctx_count + model.vocab_size
            prob = (ng_count + 1) / denom if denom > 0 else 0.0
            prob_unsmoothed = (ng_count / ctx_count) if ctx_count > 0 else 0.0
        else:
            prob = (ng_count / ctx_count) if ctx_count > 0 else 0.0
            prob_unsmoothed = prob

        if prob <= 0.0:
            had_zero = True
            logprob = -1e9
        elif logprob > -1e8:
            logprob += math.log(prob)

        rows.append(
            {
                "context": " ".join(ctx),
                "target": target,
                "count(context)": int(ctx_count),
                "count(ngram)": int(ng_count),
                "p(unsmoothed)": float(prob_unsmoothed),
                "p(used)": float(prob),
                "log p": float(math.log(prob)) if prob > 0 else float("-inf"),
            }
        )

    return {"n": model.n, "logprob": float(logprob), "had_zero": bool(had_zero), "rows": rows}
