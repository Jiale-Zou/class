from __future__ import annotations

import streamlit as st

from config import APP_TITLE, COLOR_MAP
from utils.sentiment_analyzer import analyze_text
from utils.visualization import make_confidence_gauge


def _label_badge(label: str) -> str:
    color = COLOR_MAP.get(label, COLOR_MAP["Neutral"])
    return f"<span style='padding:0.25rem 0.6rem;border-radius:999px;background:{color}22;color:{color};font-weight:600'>{label}</span>"


st.set_page_config(page_title=f"{APP_TITLE} - 显式 vs 隐式", layout="wide")
st.title("显式 vs 隐式情感识别")

col_left, col_right = st.columns(2)
with col_left:
    explicit_text = st.text_area("显式情感（直接表达）", height=140, placeholder="例如：太好了！这次购物体验非常棒。")
with col_right:
    implicit_text = st.text_area("隐式情感（事实描述暗含态度）", height=140, placeholder="例如：电池一小时就没电了。")

run = st.button("对比分析", type="primary", use_container_width=True)

if run:
    try:
        r1 = analyze_text(explicit_text)
        r2 = analyze_text(implicit_text)

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("显式结果")
            st.markdown(_label_badge(r1.label), unsafe_allow_html=True)
            st.write(f"置信度：{r1.confidence * 100:.2f}%")
            st.plotly_chart(make_confidence_gauge(r1.confidence, r1.label), use_container_width=True)
        with c2:
            st.subheader("隐式结果")
            st.markdown(_label_badge(r2.label), unsafe_allow_html=True)
            st.write(f"置信度：{r2.confidence * 100:.2f}%")
            st.plotly_chart(make_confidence_gauge(r2.confidence, r2.label), use_container_width=True)

        with st.expander("显式/隐式科普", expanded=False):
            st.write("显式情感通常包含情绪词（好、差、喜欢、不满）。")
            st.write("隐式情感常通过事实或结果描述传递态度（续航短、包装破损、发货慢）。")
            st.write("同一模型对两类表达的敏感度可能不同，可通过置信度与标签变化观察。")
    except Exception as e:
        st.error(str(e))
