from __future__ import annotations

from pathlib import Path

import streamlit as st

from config import APP_TITLE


def _load_css() -> None:
    css_path = Path(__file__).parent / "assets" / "style.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="bar_chart", layout="wide")
    _load_css()

    st.title(APP_TITLE)
    st.write("左侧侧边栏可切换功能页面：单文本分析、显式/隐式对比、舆情仪表盘。")

    with st.expander("模型与可视化说明", expanded=True):
        st.write(
            "情感分析基于 Hugging Face Transformers 的轻量级分类模型；"
            "可视化使用 Plotly（仪表盘、饼图、柱状图）。"
        )


if __name__ == "__main__":
    main()
