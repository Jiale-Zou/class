from __future__ import annotations

import re
from collections import Counter
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from config import COLOR_MAP, SENTIMENT_LABELS


def make_confidence_gauge(confidence_0_1: float, sentiment_label: str) -> go.Figure:
    confidence = max(0.0, min(1.0, float(confidence_0_1)))
    value = confidence * 100.0
    label = sentiment_label if sentiment_label in COLOR_MAP else "Neutral"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": "%"},
            title={"text": f"情感置信度: {label}"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": COLOR_MAP.get(label, COLOR_MAP["Neutral"])},
                "steps": [
                    {"range": [0, 33], "color": "rgba(239, 68, 68, 0.20)"},
                    {"range": [33, 66], "color": "rgba(234, 179, 8, 0.20)"},
                    {"range": [66, 100], "color": "rgba(34, 197, 94, 0.20)"},
                ],
            },
        )
    )
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=45, b=10))
    return fig


def make_sentiment_pie(counts: Dict[str, int]) -> go.Figure:
    labels = [lbl for lbl in SENTIMENT_LABELS if lbl in counts]
    values = [int(counts.get(lbl, 0)) for lbl in labels]
    colors = [COLOR_MAP.get(lbl, "#999999") for lbl in labels]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                marker={"colors": colors},
                hole=0.45,
            )
        ]
    )
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=35, b=10), title="整体情感分布")
    return fig


def make_confidence_bar(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return go.Figure()

    tmp = (
        df.groupby("sentiment", as_index=False)["confidence"]
        .mean()
        .rename(columns={"confidence": "avg_confidence"})
    )
    tmp["avg_confidence_pct"] = tmp["avg_confidence"] * 100.0

    fig = px.bar(
        tmp,
        x="sentiment",
        y="avg_confidence_pct",
        color="sentiment",
        color_discrete_map=COLOR_MAP,
        text="avg_confidence_pct",
        title="平均置信度（按情感）",
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=45, b=10), yaxis_title="%")
    fig.update_yaxes(range=[0, 100])
    return fig


_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]{2,}|[A-Za-z]{3,}")


def extract_keywords(texts: Iterable[str], top_k: int = 30) -> List[Tuple[str, int]]:
    counter: Counter[str] = Counter()
    for t in texts:
        if not isinstance(t, str):
            continue
        for tok in _TOKEN_RE.findall(t):
            token = tok.lower()
            if len(token) < 2:
                continue
            counter[token] += 1
    return counter.most_common(int(top_k))


def make_keyword_bar(keywords: List[Tuple[str, int]]) -> go.Figure:
    if not keywords:
        return go.Figure()
    words = [w for w, _ in keywords][::-1]
    counts = [c for _, c in keywords][::-1]
    fig = go.Figure(data=[go.Bar(x=counts, y=words, orientation="h")])
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=45, b=10), title="高频关键词（替代词云）")
    return fig
