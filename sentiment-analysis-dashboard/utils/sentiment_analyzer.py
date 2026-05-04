from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from config import HF_MODEL_NAME, MAX_TEXT_LENGTH, NEUTRAL_THRESHOLD


@dataclass(frozen=True)
class SentimentResult:
    label: str
    confidence: float
    raw_label: str
    raw_score: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "confidence": float(self.confidence),
            "raw_label": self.raw_label,
            "raw_score": float(self.raw_score),
        }


def _pick_first_item(output: Any) -> Dict[str, Any]:
    if isinstance(output, list):
        if not output:
            raise ValueError("Model returned empty output.")
        first = output[0]
        if isinstance(first, list):
            if not first:
                raise ValueError("Model returned empty top_k output.")
            return first[0]
        if isinstance(first, dict):
            return first
    if isinstance(output, dict):
        return output
    raise TypeError(f"Unexpected model output type: {type(output)}")


def _normalize_label(raw_label: str, id2label: Optional[Dict[int, str]] = None) -> str:
    label = (raw_label or "").strip()
    lower = label.lower()

    if lower in {"positive", "pos", "5 stars", "label_2"}:
        return "Positive"
    if lower in {"neutral", "neu", "3 stars", "label_1"}:
        return "Neutral"
    if lower in {"negative", "neg", "1 star", "label_0"}:
        return "Negative"

    if "pos" in lower or "positive" in lower:
        return "Positive"
    if "neu" in lower or "neutral" in lower:
        return "Neutral"
    if "neg" in lower or "negative" in lower:
        return "Negative"

    if id2label:
        reverse = {str(v).lower(): k for k, v in id2label.items()}
        if lower in reverse:
            mapped = str(id2label[int(reverse[lower])]).lower()
            if "pos" in mapped:
                return "Positive"
            if "neu" in mapped:
                return "Neutral"
            if "neg" in mapped:
                return "Negative"

    return "Neutral"


@lru_cache(maxsize=1)
def get_sentiment_pipeline(model_name: str = HF_MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return pipeline(
        task="sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        truncation=True,
        max_length=MAX_TEXT_LENGTH,
    )


def analyze_text(text: str, model_name: str = HF_MODEL_NAME) -> SentimentResult:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("请输入非空文本。")

    pipe = get_sentiment_pipeline(model_name=model_name)
    out = pipe(text)
    first = _pick_first_item(out)

    raw_label = str(first.get("label", ""))
    raw_score = float(first.get("score", 0.0))

    id2label = None
    model = getattr(pipe, "model", None)
    config = getattr(model, "config", None)
    if config is not None and getattr(config, "id2label", None):
        try:
            id2label = {int(k): str(v) for k, v in dict(config.id2label).items()}
        except Exception:
            id2label = None

    normalized = _normalize_label(raw_label, id2label=id2label)

    num_labels = getattr(config, "num_labels", None) if config is not None else None
    if num_labels == 2 and raw_score < NEUTRAL_THRESHOLD:
        normalized = "Neutral"

    return SentimentResult(
        label=normalized,
        confidence=raw_score,
        raw_label=raw_label,
        raw_score=raw_score,
    )


def analyze_batch(texts: Sequence[str], model_name: str = HF_MODEL_NAME) -> List[SentimentResult]:
    pipe = get_sentiment_pipeline(model_name=model_name)
    out = pipe(list(texts))

    results: List[SentimentResult] = []
    model = getattr(pipe, "model", None)
    config = getattr(model, "config", None)
    id2label = None
    if config is not None and getattr(config, "id2label", None):
        try:
            id2label = {int(k): str(v) for k, v in dict(config.id2label).items()}
        except Exception:
            id2label = None

    num_labels = getattr(config, "num_labels", None) if config is not None else None

    if not isinstance(out, list):
        out = [out]

    for item in out:
        first = _pick_first_item(item)
        raw_label = str(first.get("label", ""))
        raw_score = float(first.get("score", 0.0))
        normalized = _normalize_label(raw_label, id2label=id2label)
        if num_labels == 2 and raw_score < NEUTRAL_THRESHOLD:
            normalized = "Neutral"
        results.append(
            SentimentResult(
                label=normalized,
                confidence=raw_score,
                raw_label=raw_label,
                raw_score=raw_score,
            )
        )

    return results
