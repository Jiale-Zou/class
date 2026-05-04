from __future__ import annotations

import pandas as pd
import streamlit as st

from config import APP_TITLE, DEFAULT_DASHBOARD_ROWS, SENTIMENT_LABELS
from utils.data_generator import generate_mock_comments
from utils.sentiment_analyzer import analyze_batch
from utils.visualization import (
    extract_keywords,
    make_confidence_bar,
    make_keyword_bar,
    make_sentiment_pie,
)


@st.cache_data(show_spinner=False)
def _mock_data(n: int) -> pd.DataFrame:
    return generate_mock_comments(n=n)


@st.cache_data(show_spinner=False)
def _analyze(df: pd.DataFrame) -> pd.DataFrame:
    results = analyze_batch(df["text"].tolist())
    out = df.copy()
    out["sentiment"] = [r.label for r in results]
    out["confidence"] = [r.confidence for r in results]
    return out


st.set_page_config(page_title=f"{APP_TITLE} - 舆情仪表盘", layout="wide")
st.title("舆情挖掘与可视化仪表盘")

with st.sidebar:
    st.subheader("数据配置")
    n = st.slider("模拟评论数量", min_value=20, max_value=300, value=DEFAULT_DASHBOARD_ROWS, step=10)
    refresh = st.button("重新生成并分析", use_container_width=True)

if refresh:
    _mock_data.clear()
    _analyze.clear()

df0 = _mock_data(n)

with st.spinner("批量情感分析中…"):
    df = _analyze(df0)

counts = df["sentiment"].value_counts().to_dict()
pos = int(counts.get("Positive", 0))
neu = int(counts.get("Neutral", 0))
neg = int(counts.get("Negative", 0))

m1, m2, m3, m4 = st.columns(4)
m1.metric("好评", pos)
m2.metric("中评", neu)
m3.metric("差评", neg)
m4.metric("总量", int(len(df)))

c1, c2, c3 = st.columns([1.1, 1.1, 1.0])
with c1:
    st.plotly_chart(make_sentiment_pie(counts), use_container_width=True)
with c2:
    st.plotly_chart(make_confidence_bar(df), use_container_width=True)
with c3:
    keywords = extract_keywords(df["text"].tolist(), top_k=25)
    st.plotly_chart(make_keyword_bar(keywords), use_container_width=True)

st.subheader("详细评论列表")
view_cols = ["created_at", "platform", "category", "text", "sentiment", "confidence"]
st.dataframe(
    df[view_cols].sort_values("created_at", ascending=False),
    use_container_width=True,
    hide_index=True,
)
