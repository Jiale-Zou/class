from __future__ import annotations

import pandas as pd
import streamlit as st

from graph_viz import LABEL_COLOR, build_edges, build_nodes, ensure_nodes_for_relations, render_vis_network_html
from mock_data import mock_ner
from ner import TokenSpan, bio_tag, format_bio, render_highlighted_html, spacy_bio, spacy_entities, spacy_tokens, tokenize_with_offsets
from re_extraction import extract_relations


APP_TITLE = "NLP 信息抽取与知识图谱构建演示"


def _example_texts() -> dict[str, str]:
    return {
        "英文示例": "Steve Jobs founded Apple in California. He later joined Apple again.",
        "中文示例": "乔布斯在加州创立了苹果公司。他后来加入苹果。",
    }


def _label_colors() -> dict[str, str]:
    return {
        "PERSON": LABEL_COLOR["PERSON"],
        "ORG": LABEL_COLOR["ORG"],
        "LOC": LABEL_COLOR["LOC"],
    }


def _detect_lang(text: str) -> str:
    for ch in text:
        if "\u4e00" <= ch <= "\u9fff":
            return "zh"
    return "en"


@st.cache_resource(show_spinner=False)
def _get_spacy_nlp(lang: str):
    import spacy

    model = "zh_core_web_sm" if lang == "zh" else "en_core_web_sm"
    try:
        return spacy.load(model), []
    except Exception:
        warnings = [
            f"未能加载 spaCy 模型 {model}。可在终端执行：python -m spacy download {model}",
        ]
        try:
            from spacy.cli import download

            download(model)
            return spacy.load(model), warnings
        except Exception:
            warnings.append("自动下载失败，已回退到空白分词器（将无法识别实体/关系）。")
            return spacy.blank("zh" if lang == "zh" else "en"), warnings



def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    with st.sidebar:
        st.subheader("输入")
        examples = _example_texts()
        example_choice = st.selectbox("示例句", options=["自定义"] + list(examples.keys()))

        default_text = ""
        if example_choice != "自定义":
            default_text = examples[example_choice]

        text = st.text_area("文本输入", value=default_text, height=180, placeholder="粘贴或输入任意中英文文本…")
        show_bio = st.checkbox("查看底层 BIO 标注", value=False)
        extraction_mode = st.selectbox("抽取模式", options=["spaCy", "Mock"], index=0)
        lang_choice = st.selectbox("语言", options=["自动", "英文", "中文"], index=0)

        run = st.button("抽取", type="primary", use_container_width=True)

    if "result" not in st.session_state:
        st.session_state.result = None

    if run:
        if not text.strip():
            st.warning("请输入文本后再抽取。")
            st.session_state.result = None
        else:
            warnings: list[str] = []
            doc = None
            if extraction_mode == "spaCy":
                if lang_choice == "英文":
                    lang = "en"
                elif lang_choice == "中文":
                    lang = "zh"
                else:
                    lang = _detect_lang(text)

                nlp, nlp_warnings = _get_spacy_nlp(lang)
                warnings.extend(nlp_warnings)
                doc = nlp(text)

                entities = spacy_entities(doc)
                tokens = spacy_tokens(doc)
                bio = spacy_bio(doc)
                relations, re_warnings = extract_relations(text, entities, mode="spacy", doc=doc)
                warnings.extend(re_warnings)
            else:
                entities = mock_ner(text)
                tokens = tokenize_with_offsets(text)
                bio = bio_tag(tokens, entities)
                relations, re_warnings = extract_relations(text, entities, mode="mock")
                warnings.extend(re_warnings)

            st.session_state.result = {
                "text": text,
                "entities": entities,
                "tokens": [{"token": t.token, "start": t.start, "end": t.end} for t in tokens],
                "bio_tags": bio,
                "relations": relations,
                "warnings": warnings,
            }

    result = st.session_state.result
    if not result:
        st.info("在左侧输入文本并点击“抽取”，即可看到实体、关系与知识图谱。")
        return

    tokens = [TokenSpan(token=t["token"], start=t["start"], end=t["end"]) for t in result["tokens"]]
    entities = result["entities"]
    bio = result["bio_tags"]
    relations = result["relations"]

    if len(tokens) > 500:
        st.warning("文本较长（>500 tokens），展示与可视化可能变慢。建议先用短文本做教学演示。")

    left, right = st.columns([1, 1], gap="large")
    with left:
        st.subheader("实体（NER）")
        if not entities:
            st.warning("未识别到实体。")
        else:
            highlighted = render_highlighted_html(result["text"], entities, _label_colors())
            st.markdown(highlighted, unsafe_allow_html=True)
            legend = " ".join(
                f'<span style="background-color:{c}; padding:0 6px; border-radius:10px;">{lbl}</span>'
                for lbl, c in _label_colors().items()
            )
            st.markdown(legend, unsafe_allow_html=True)

        if show_bio:
            st.subheader("BIO 标注（底层序列）")
            st.code(format_bio(tokens, bio), language="text")

    with right:
        st.subheader("知识图谱")
        if result.get("warnings"):
            for w in result["warnings"]:
                st.info(w)

        nodes = build_nodes(entities)
        edges = build_edges(relations)
        nodes = ensure_nodes_for_relations(nodes, edges)
        html = render_vis_network_html(nodes, edges, height_px=600)
        st.components.v1.html(html, height=620)

        if not relations:
            st.warning("未抽取到关系（当前图谱仅显示实体节点）。")

        st.subheader("关系表（Subject–Predicate–Object）")
        if relations:
            df = pd.DataFrame(relations)
            q = st.text_input("关系搜索", value="", placeholder="输入关键词过滤（如 Apple / FOUNDED_BY）")
            if q.strip():
                q_norm = q.strip().lower()
                mask = df.apply(lambda row: q_norm in " ".join(map(str, row.values)).lower(), axis=1)
                df = df[mask]
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("关系表将在抽取到三元组后显示。")

    with st.expander("学生观察与思考题", expanded=False):
        st.markdown(
            """
- Q1：BIO 标注如何帮助模型理解实体边界？
- Q2：嵌套实体（如 “University of California, Los Angeles”）在单层 BIO 中会出现什么问题？
- Q3：关系抽取与知识图谱边的关系是什么？
- Q4：从文本到图谱，信息发生了哪些结构性变化？
"""
        )


if __name__ == "__main__":
    main()

