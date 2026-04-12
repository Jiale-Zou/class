from __future__ import annotations

import csv
import io
import json
import sys
from pathlib import Path
from typing import List

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import connectives as conn
from utils import coref_viz as cv
from utils import edu_segmentation as edu

# Optional: register fastcoref factory on import
try:
    import fastcoref.spacy_component  # noqa: F401
except Exception:
    pass


st.set_page_config(page_title="篇章分析三合一", layout="wide", initial_sidebar_state="expanded")


@st.cache_resource(show_spinner=True)
def load_nlp_edu():
    import spacy
    from spacy.cli import download as spacy_download

    try:
        return spacy.load("en_core_web_sm")
    except Exception:
        spacy_download("en_core_web_sm")
        return spacy.load("en_core_web_sm")


@st.cache_resource(show_spinner=True)
def load_nlp_coref():
    return cv.try_load_coref_nlp()


def ensure_samples() -> List[str]:
    ok: List[str] = []
    for name in edu.SAMPLE_FILES:
        try:
            edu.fetch_sample_pair(name)
            ok.append(name)
        except Exception:
            continue
    return ok or ["wsj_0601.out"]


def tab_edu(nlp):
    st.subheader("模块1：话语分割（EDU）规则基线 vs NeuralEDUSeg 标注")
    st.caption(
        "数据来自 PKU-TANGENT/NeuralEDUSeg（`data/rst/TRAINING/*.out.edus`，一行一条 EDU）。"
        "规则基线基于 spaCy `en_core_web_sm` 依存与词性启发式。"
    )

    samples = ensure_samples()
    c1, c2 = st.columns(2)
    with c1:
        sample = st.selectbox("样本文件", samples, index=0)
    with c2:
        max_edus = st.slider("最多展示 EDU 条数", min_value=3, max_value=40, value=14)

    try:
        out_raw, edus_raw = edu.fetch_sample_pair(sample)
    except Exception as exc:
        st.error(f"拉取样本失败：{exc}")
        return

    xml_segs = edu.parse_edu_xml(edus_raw)
    if xml_segs is not None:
        segments = xml_segs[:max_edus]
        st.info("检测到 `<EDU>` 标签格式，已按标签解析。")
    else:
        segments = edu.parse_edu_lines(edus_raw)[:max_edus]

    canonical = edu.canonical_from_edus(segments)

    doc = nlp(canonical)
    flags = edu.RuleFlags(
        punct=st.checkbox("标点边界 (. ? !)", True, key="f_punct"),
        mark_cc=st.checkbox("mark / cc 边界", True, key="f_mc"),
        sconj=st.checkbox("从属连词 SCONJ 边界", True, key="f_sc"),
        root=st.checkbox("ROOT 边界（保守，默认关）", False, key="f_rt"),
    )
    hl = st.checkbox("边界首词高亮（黄色）", True, key="edu_hl")
    show_diff = st.checkbox("差异颜色标记（预测有/金标无 与 反之）", True, key="edu_diff")

    pred_b = edu.predict_boundaries(doc, flags)
    starts_chars = edu.char_starts_for_segments(segments)
    gold_b = edu.gold_boundaries_before_tokens(doc, starts_chars)

    n_tok = len(doc)
    p, r, f1 = edu.boundary_f1(pred_b, gold_b, n_tok)

    pred_seg = edu.boundaries_to_segments(doc, pred_b)
    gold_seg = list(segments)

    st.markdown("#### 统计")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Precision", f"{p:.3f}")
    m2.metric("Recall", f"{r:.3f}")
    m3.metric("F1（边界）", f"{f1:.3f}")
    m4.metric("预测 EDU 数 / 金标", f"{len(pred_seg)} / {len(gold_seg)}")

    rows = []
    inner = set(range(1, max(1, n_tok)))
    for j in sorted(pred_b & inner):
        lab = edu.explain_boundary(doc, j, flags)
        rows.append(
            {
                "boundary_before_token": j,
                "rule": lab,
                "hit": int(j in gold_b),
            }
        )
    if rows:
        st.markdown("#### 边界类型 × 是否命中金标")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        bc = pd.DataFrame(rows).groupby("rule")["hit"].mean().reset_index()
        bc.columns = ["rule", "mean_hit"]
        st.bar_chart(bc.set_index("rule"))

    lens = edu.per_segment_length_analysis(gold_seg, pred_seg)
    if lens:
        st.markdown("#### 长度影响（按索引粗略对齐）")
        st.dataframe(pd.DataFrame(lens), use_container_width=True, hide_index=True)

    left, right = st.columns(2)
    with left:
        st.markdown("**规则基线**")
        btok = set(pred_b)
        html_l = edu.edu_cards_html(pred_seg, border_color="#2563eb", title="规则 EDU")
        if hl:
            html_l += "<div style='margin-top:10px'>" + edu.highlight_boundary_words(doc, btok, highlight_boundaries=True) + "</div>"
        components.html(f"<div style='font-family:system-ui'>{html_l}</div>", height=520, scrolling=True)
    with right:
        st.markdown("**NeuralEDUSeg 金标**")
        g_starts = sorted({0} | set(gold_b))
        gtok = set(gold_b)
        html_r = edu.edu_cards_html(gold_seg, border_color="#16a34a", title="金标 EDU")
        if hl:
            html_r += "<div style='margin-top:10px'>" + edu.highlight_boundary_words(doc, gtok, highlight_boundaries=True) + "</div>"
        components.html(f"<div style='font-family:system-ui'>{html_r}</div>", height=520, scrolling=True)

    if show_diff:
        fp = pred_b - gold_b
        fn = gold_b - pred_b
        st.markdown("#### 差异（token 级：边界位于「该 token 之前」）")
        st.write(
            {
                "FP（预测有、金标无）": sorted(fp),
                "FN（金标有、预测无）": sorted(fn),
            }
        )

    export = {
        "sample": sample,
        "canonical_prefix": canonical,
        "flags": flags.__dict__,
        "precision": p,
        "recall": r,
        "f1": f1,
        "pred_boundaries": sorted(pred_b),
        "gold_boundaries": sorted(gold_b),
        "pred_segments": pred_seg,
        "gold_segments": gold_seg,
    }
    st.download_button(
        "导出 JSON",
        data=json.dumps(export, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name="edu_compare.json",
        mime="application/json",
    )


def tab_connectives(nlp):
    st.subheader("模块2：浅层篇章分析 / 显式连接词与论据")
    default_txt = (
        "Third-quarter sales in Europe were exceptionally strong, boosted by promotional programs "
        "and new products - although weaker foreign currencies reduced the company's earnings."
    )
    text = st.text_area("分析文本", value=default_txt, height=140)
    up = st.file_uploader("批量上传 .txt（多文件拼接）", type=["txt"], accept_multiple_files=True)
    if up:
        chunks = []
        for f in up:
            chunks.append(f.read().decode("utf-8", errors="replace"))
        text = "\n\n".join(chunks)

    doc = nlp(text)
    hits = conn.scan_connectives(doc)

    st.markdown("#### 连接词库（语义类）")
    st.code(json.dumps(conn.CONNECTIVES, ensure_ascii=False, indent=2), language="json")

    st.markdown("#### 标注视图")
    html = conn.render_connective_html(doc, hits)
    components.html(f"<div style='font-family:Georgia,serif;font-size:17px;line-height:1.6'>{html}</div>", height=220, scrolling=True)

    st.markdown("#### 匹配结果")
    if not hits:
        st.info("未命中连接词库中的显式连接成分。")
    else:
        st.dataframe(
            pd.DataFrame([h.__dict__ for h in hits]),
            use_container_width=True,
            hide_index=True,
        )
        for h in hits:
            with st.expander(f"{h.surface} → {h.category}"):
                st.markdown(conn.render_args_blocks(h), unsafe_allow_html=True)

    st.markdown("#### since 消歧测试")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("运行：TEMPORAL 例句"):
            st.session_state["since_a"] = "I've been working here since 2010."
    with c2:
        if st.button("运行：CONTINGENCY 例句"):
            st.session_state["since_b"] = "Since it was raining, we stayed home."
    sa = st.session_state.get("since_a", "I've been working here since 2010.")
    sb = st.session_state.get("since_b", "Since it was raining, we stayed home.")
    d1, d2 = nlp(sa), nlp(sb)
    h1, h2 = conn.scan_connectives(d1), conn.scan_connectives(d2)
    st.write(
        {
            "A": {"text": sa, "hits": [x.__dict__ for x in h1]},
            "B": {"text": sb, "hits": [x.__dict__ for x in h2]},
        }
    )

    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["category", "surface", "arg1", "arg2", "note"])
    for h in hits:
        w.writerow([h.category, h.surface, h.arg1, h.arg2, h.note])
    st.download_button("导出连接词 CSV", data=buf.getvalue().encode("utf-8-sig"), file_name="connectives.csv", mime="text/csv")


def tab_coref():
    st.subheader("模块3：指代消解可视化（fastcoref）")
    nlp_c, err = load_nlp_coref()
    if nlp_c is None:
        st.error(f"无法加载 fastcoref 管道：{err}")
        st.info("请确认已安装 `torch` 与 `fastcoref`，且可下载 HuggingFace 权重。")
        return

    default = (
        "Barack Obama visited Berlin. He spoke about climate policy. "
        "Michelle Obama joined him later. She attended a youth forum."
    )
    text = st.text_area("段落输入", value=default, height=160)
    lim = st.number_input("最大字符数", min_value=200, max_value=8000, value=2500, step=100)
    if len(text) > lim:
        text = text[:lim]
        st.caption(f"文本已截断至 {lim} 字符。")

    try:
        with st.spinner("fastcoref 推理中…"):
            _doc, clusters = cv.run_coref(nlp_c, text)
    except Exception as exc:
        st.exception(exc)
        return

    if not clusters:
        st.warning("未检测到指代簇（可能模型未返回结果）。")

    html = cv.render_coref_html(text, clusters)
    components.html(f"<div style='font-family:system-ui'>{html}</div>", height=320, scrolling=True)

    rows = cv.clusters_to_rows(clusters, text)
    st.markdown("#### 簇结构表")
    flat = []
    for row in rows:
        for m in row["mentions"]:
            flat.append(
                {
                    "cluster_id": row["cluster_id"],
                    "start": m["start"],
                    "end": m["end"],
                    "text": m["text"],
                }
            )
    st.dataframe(pd.DataFrame(flat), use_container_width=True, hide_index=True)

    st.markdown("#### 结构化输出（示意）")
    lines = []
    for row in rows:
        parts = []
        for m in row["mentions"]:
            parts.append(f"{m['text']} ({m['start']}-{m['end']})")
        lines.append(f"Cluster {row['cluster_id']}: " + ", ".join(parts))
    st.code("\n".join(lines), language="text")

    st.caption("置信度：fastcoref 的 spaCy 组件不直接暴露逐边分数；可在 Python 端用 FCoref.predict 获取 logits。")
    st.caption("截图：请使用系统或浏览器截图工具保存本页视图。")

    st.download_button(
        "导出指代 JSON",
        data=json.dumps({"text": text, "clusters": clusters}, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name="coref.json",
        mime="application/json",
    )


def main():
    st.title("篇章分析演示：EDU 切分 / 显式关系 / 指代消解")
    with st.sidebar:
        st.markdown("### 运行说明")
        st.markdown(
            "- 首次运行会下载 **en_core_web_sm** 与 **fastcoref** 权重。\n"
            "- 若指代模块报 `all_tied_weights_keys`：请将 **transformers 固定为 4.x**（勿使用 5.x，见 requirements.txt）。\n"
            "- 模块1 会缓存 NeuralEDUSeg 原始文件到 `discourse-analysis-app/data/cache/`。\n"
            "- 若 GPU 可用，可在代码中将 fastcoref 配置为 `cuda`。"
        )

    nlp = load_nlp_edu()

    tab1, tab2, tab3 = st.tabs(
        [
            "📊 话语分割对比",
            "🔗 显式关系分析",
            "🎯 指代消解可视化",
        ]
    )
    with tab1:
        tab_edu(nlp)
    with tab2:
        tab_connectives(nlp)
    with tab3:
        tab_coref()


if __name__ == "__main__":
    main()
