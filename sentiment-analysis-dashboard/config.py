from __future__ import annotations

import os

APP_TITLE = os.getenv("APP_TITLE", "舆情分析工具")

HF_MODEL_NAME = os.getenv("HF_MODEL_NAME", "uer/roberta-base-finetuned-jd-binary-chinese")
NEUTRAL_THRESHOLD = float(os.getenv("NEUTRAL_THRESHOLD", "0.60"))

MAX_TEXT_LENGTH = int(os.getenv("MAX_TEXT_LENGTH", "512"))

DEFAULT_DASHBOARD_ROWS = int(os.getenv("DEFAULT_DASHBOARD_ROWS", "60"))
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))

SENTIMENT_LABELS = ("Positive", "Neutral", "Negative")

COLOR_MAP = {
    "Positive": "#22c55e",
    "Neutral": "#eab308",
    "Negative": "#ef4444",
}
