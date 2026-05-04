from __future__ import annotations

import streamlit as st

from config import APP_TITLE, COLOR_MAP
from utils.sentiment_analyzer import analyze_text
from utils.visualization import make_confidence_gauge


def _label_badge(label: str) -> str:
    color = COLOR_MAP.get(label, COLOR_MAP["Neutral"])
    return f"<span style='padding:0.25rem 0.6rem;border-radius:999px;background:{color}22;color:{color};font-weight:600'>{label}</span>"


st.set_page_config(page_title=f"{APP_TITLE} - 单文本分析", layout="wide")
st.title("单文本情感分析")

text = st.text_area("输入文本", height=140, placeholder="例如：质量很好，物流也很快。")
run = st.button("分析", type="primary", use_container_width=True)

if run:
    try:
        result = analyze_text(text)
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("分析结果")
            st.markdown(_label_badge(result.label), unsafe_allow_html=True)
            st.write(f"置信度：{result.confidence * 100:.2f}%")
        with col2:
            st.plotly_chart(make_confidence_gauge(result.confidence, result.label), use_container_width=True)
    except Exception as e:
        st.error(str(e))
