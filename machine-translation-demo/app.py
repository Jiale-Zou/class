from __future__ import annotations

import json
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import streamlit as st

from config import DEFAULT_CONFIG
from utils import bleu_score_with_details
from utils.display_helpers import block_html, word_diff_spans
from utils.rule_based_translator import RuleBasedTranslator, load_dictionary


BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"


def _safe_read_text(path: Path, limit_chars: int = 20000) -> str:
    try:
        return path.read_text(encoding="utf-8")[:limit_chars]
    except Exception:
        return ""


def _load_examples(examples_dir: Path) -> Dict[str, str]:
    if not examples_dir.exists() or not examples_dir.is_dir():
        return {
            "短句": "Hello world!",
            "日常对话": "Good morning. Thank you for your help!",
            "技术文本": "Machine translation models learn to map source sentences to target sentences.",
            "长一点": (
                "In recent years, neural machine translation has achieved impressive results. "
                "However, errors such as mistranslation and omission can still happen, "
                "especially on long sentences and domain-specific terms."
            ),
        }
    out: Dict[str, str] = {}
    for p in sorted(examples_dir.glob("*.txt")):
        name = p.stem
        content = _safe_read_text(p).strip()
        if content:
            out[name] = content
    return out or {
        "短句": "Hello world!",
        "技术文本": "Machine translation models learn to map source sentences to target sentences.",
    }


def _ensure_nltk_sentence_splitter() -> None:
    try:
        import nltk

        try:
            nltk.data.find("tokenizers/punkt")
        except Exception:
            try:
                nltk.download("punkt", quiet=True)
            except Exception:
                pass
    except Exception:
        return


def _split_sentences(text: str) -> List[str]:
    t = (text or "").strip()
    if not t:
        return []
    _ensure_nltk_sentence_splitter()
    try:
        from nltk.tokenize import sent_tokenize

        sents = [s.strip() for s in sent_tokenize(t) if s.strip()]
        return sents if sents else [t]
    except Exception:
        parts = []
        buf = []
        for ch in t:
            buf.append(ch)
            if ch in ".!?。！？":
                s = "".join(buf).strip()
                if s:
                    parts.append(s)
                buf = []
        tail = "".join(buf).strip()
        if tail:
            parts.append(tail)
        return parts or [t]


def _make_chunks(sentences: List[str], chunk_chars: int) -> List[str]:
    out: List[str] = []
    cur: List[str] = []
    cur_len = 0
    for s in sentences:
        s_len = len(s)
        if cur and cur_len + 1 + s_len > chunk_chars:
            out.append(" ".join(cur).strip())
            cur = [s]
            cur_len = s_len
        else:
            cur.append(s)
            cur_len = cur_len + (1 if cur_len else 0) + s_len
    if cur:
        out.append(" ".join(cur).strip())
    return [c for c in out if c]


@st.cache_resource(show_spinner=False)
def _get_translation_pipeline(model_name: str, device_preference: str):
    import torch
    from transformers import pipeline

    device = -1
    if device_preference == "GPU" and torch.cuda.is_available():
        device = 0
    return pipeline("translation", model=model_name, device=device)


def _translate_with_progress(
    model_name: str,
    device_preference: str,
    text: str,
    max_new_tokens: int,
    chunk_chars: int,
) -> Tuple[Optional[str], Optional[str]]:
    t0 = time.perf_counter()
    try:
        pipe = _get_translation_pipeline(model_name, device_preference)
    except Exception as exc:
        return None, f"模型加载失败：{exc}"

    sents = _split_sentences(text)
    chunks = _make_chunks(sents, max(100, int(chunk_chars)))
    if not chunks:
        return "", None

    prog = st.progress(0.0)
    status = st.empty()
    outputs: List[str] = []
    for i, chunk in enumerate(chunks, start=1):
        status.info(f"正在翻译：{i}/{len(chunks)}")
        try:
            r = pipe(chunk, max_new_tokens=int(max_new_tokens))
            if isinstance(r, list) and r and isinstance(r[0], dict) and "translation_text" in r[0]:
                outputs.append(str(r[0]["translation_text"]).strip())
            else:
                outputs.append(str(r))
        except Exception as exc:
            return None, f"翻译失败：{exc}"
        prog.progress(i / len(chunks))

    status.success(f"完成，用时 {int((time.perf_counter() - t0) * 1000)} ms")
    return "\n".join([o for o in outputs if o]).strip(), None


_EXECUTOR = ThreadPoolExecutor(max_workers=1)


def _submit_translation_job(params: dict) -> None:
    st.session_state["nmt_job_params"] = params
    st.session_state["nmt_job_future"] = _EXECUTOR.submit(_translate_worker, params)


def _translate_worker(params: dict) -> dict:
    import torch
    from transformers import pipeline

    model_name = str(params["model_name"])
    device_preference = str(params["device_preference"])
    max_new_tokens = int(params["max_new_tokens"])
    chunk_chars = int(params["chunk_chars"])
    text = str(params["text"])

    device = -1
    if device_preference == "GPU" and torch.cuda.is_available():
        device = 0
    pipe = pipeline("translation", model=model_name, device=device)

    sents = _split_sentences(text)
    chunks = _make_chunks(sents, max(100, chunk_chars))
    outputs: List[str] = []
    for chunk in chunks:
        r = pipe(chunk, max_new_tokens=max_new_tokens)
        if isinstance(r, list) and r and isinstance(r[0], dict) and "translation_text" in r[0]:
            outputs.append(str(r[0]["translation_text"]).strip())
        else:
            outputs.append(str(r))
    return {"translation": "\n".join([o for o in outputs if o]).strip()}


def _poll_translation_job() -> Tuple[Optional[str], Optional[str], bool]:
    fut = st.session_state.get("nmt_job_future")
    if fut is None:
        return None, None, False
    if not fut.done():
        return None, None, True
    try:
        payload = fut.result()
        return str(payload.get("translation", "")), None, False
    except Exception as exc:
        return None, f"后台翻译失败：{exc}", False


def _render_sidebar() -> dict:
    st.sidebar.header("设置")
    model_name = st.sidebar.text_input("模型", value=DEFAULT_CONFIG.model_name)
    max_new_tokens = st.sidebar.slider("最大生成长度（tokens）", min_value=16, max_value=512, value=DEFAULT_CONFIG.max_new_tokens, step=8)
    chunk_chars = st.sidebar.slider("分块大小（字符）", min_value=200, max_value=2000, value=DEFAULT_CONFIG.chunk_chars, step=50)
    device_preference = st.sidebar.radio("运行设备", options=["CPU", "GPU"], index=1 if DEFAULT_CONFIG.use_gpu_if_available else 0)
    highlight = st.sidebar.checkbox("对比模块：高亮差异", value=True)
    smooth_bleu = st.sidebar.checkbox("BLEU：平滑处理", value=True)
    max_n = st.sidebar.select_slider("BLEU：最大 n", options=[1, 2, 3, 4], value=4)
    return {
        "model_name": model_name.strip() or DEFAULT_CONFIG.model_name,
        "max_new_tokens": int(max_new_tokens),
        "chunk_chars": int(chunk_chars),
        "device_preference": device_preference,
        "highlight_diff": bool(highlight),
        "bleu_smooth": bool(smooth_bleu),
        "bleu_max_n": int(max_n),
    }


def _tab_nmt(settings: dict, examples: Dict[str, str]) -> None:
    st.subheader("神经网络翻译")

    c1, c2 = st.columns([2, 1])
    with c2:
        example_key = st.selectbox("示例文本", options=["(不选择)"] + list(examples.keys()), index=0)
        use_example = st.button("填入示例", use_container_width=True)

    default_text = ""
    if use_example and example_key in examples:
        default_text = examples[example_key]

    text = st.text_area(
        "英文输入",
        value=default_text,
        height=220,
        placeholder="输入英文文本，支持多行",
        key="nmt_input",
    )

    words = [w for w in (text or "").split() if w.strip()]
    st.caption(f"字数：{len(words)}")

    col_a, col_b, col_c = st.columns([1, 1, 2])
    with col_a:
        run = st.button("翻译", type="primary", use_container_width=True)
    with col_b:
        stop = st.button("停止后台任务", use_container_width=True)
    with col_c:
        st.caption("后台模式：提交后可切换标签页，完成后会在本标签显示结果。")

    if stop:
        st.session_state.pop("nmt_job_future", None)
        st.session_state.pop("nmt_job_params", None)
        st.session_state.pop("nmt_last_result", None)
        st.warning("已清空当前后台任务状态（线程可能仍在执行，但结果不会再写入界面）。")

    if run and (text or "").strip():
        params = {
            "model_name": settings["model_name"],
            "device_preference": settings["device_preference"],
            "max_new_tokens": settings["max_new_tokens"],
            "chunk_chars": settings["chunk_chars"],
            "text": text,
        }
        _submit_translation_job(params)

    result, err, pending = _poll_translation_job()
    if pending:
        st.info("后台翻译进行中…（页面会在你点击任意控件时刷新状态）")
    if err:
        st.error(err)
    if result is not None:
        st.session_state["nmt_last_result"] = result

    last = st.session_state.get("nmt_last_result")
    if last:
        st.markdown("翻译结果")
        st.code(last, language=None)


def _tab_compare(settings: dict, examples: Dict[str, str]) -> None:
    st.subheader("规则 vs 神经网络")

    dict_path = str((BASE_DIR / DEFAULT_CONFIG.dictionary_path).as_posix())
    dictionary = load_dictionary(dict_path)
    rule_translator = RuleBasedTranslator(dictionary=dictionary)

    c1, c2 = st.columns([2, 1])
    with c2:
        example_key = st.selectbox("典型案例", options=["(不选择)"] + list(examples.keys()), index=0, key="cmp_ex")
        use_example = st.button("使用案例", use_container_width=True, key="cmp_fill")

    default_text = ""
    if use_example and example_key in examples:
        default_text = examples[example_key]

    text = st.text_area("英文输入", value=default_text, height=180, key="cmp_input")
    if not (text or "").strip():
        st.info("输入英文文本后即可对比两种翻译。")
        return

    rule_out = rule_translator.translate(text)

    nmt_out = ""
    nmt_err = None
    with st.spinner("正在生成神经网络译文…"):
        nmt_out, nmt_err = _translate_with_progress(
            model_name=settings["model_name"],
            device_preference=settings["device_preference"],
            text=text,
            max_new_tokens=settings["max_new_tokens"],
            chunk_chars=settings["chunk_chars"],
        )
    if nmt_err:
        st.error(nmt_err)
        return

    left, right = st.columns(2)
    if settings["highlight_diff"]:
        a_html, b_html = word_diff_spans(rule_out, nmt_out)
        with left:
            st.markdown("规则翻译（差异高亮）")
            st.markdown(block_html(a_html), unsafe_allow_html=True)
        with right:
            st.markdown("神经网络翻译（差异高亮）")
            st.markdown(block_html(b_html), unsafe_allow_html=True)
    else:
        with left:
            st.markdown("规则翻译")
            st.code(rule_out, language=None)
        with right:
            st.markdown("神经网络翻译")
            st.code(nmt_out, language=None)

    with st.expander("翻译方法说明", expanded=False):
        st.write(
            "规则翻译：基于词典逐词映射 + 简单标点处理，未知词保留。"
            "神经网络翻译：使用 Hugging Face Transformers 的翻译 pipeline。"
        )


def _tab_bleu(settings: dict) -> None:
    st.subheader("翻译质量评测（BLEU）")

    c1, c2, c3 = st.columns(3)
    with c1:
        src = st.text_area("英文原文", height=150, key="bleu_src")
    with c2:
        ref = st.text_area("人工参考译文", height=150, key="bleu_ref")
    with c3:
        hyp = st.text_area("机器候选译文", height=150, key="bleu_hyp")

    col_a, col_b = st.columns([1, 2])
    with col_a:
        run = st.button("计算 BLEU", type="primary", use_container_width=True)
    with col_b:
        st.caption("BLEU 适合对比同一语料上的系统差异；单句分数波动较大。")

    if run:
        details = bleu_score_with_details(ref, hyp, max_n=settings["bleu_max_n"], smooth=settings["bleu_smooth"])
        st.session_state["bleu_last"] = details
        case = {"src": src, "ref": ref, "hyp": hyp, "settings": {"max_n": settings["bleu_max_n"], "smooth": settings["bleu_smooth"]}}
        st.session_state.setdefault("bleu_cases", [])
        st.session_state["bleu_cases"].append(case)

    details = st.session_state.get("bleu_last")
    if details is None:
        st.info("填写参考译文与候选译文后点击计算。")
        return

    st.markdown("评测结果")
    st.metric("BLEU", value=f"{details.bleu * 100:.2f}", delta=None)
    st.caption(f"BP={details.bp:.4f}，参考长度={details.ref_len}，候选长度={details.cand_len}")

    rows = []
    for n in range(1, settings["bleu_max_n"] + 1):
        p = details.precisions.get(n, 0.0)
        m = details.matches.get(n, 0)
        t = details.totals.get(n, 0)
        rows.append({"n": n, "precision": p, "match/total": f"{m}/{t}"})
    st.dataframe(rows, use_container_width=True, hide_index=True)

    chart_data = {f"{n}-gram": details.precisions.get(n, 0.0) for n in range(1, settings["bleu_max_n"] + 1)}
    st.bar_chart(chart_data)

    with st.expander("匹配示例（Top n-grams）", expanded=False):
        for n in range(1, settings["bleu_max_n"] + 1):
            ex = details.matched_ngrams_examples.get(n, [])
            if not ex:
                continue
            st.markdown(f"{n}-gram")
            st.write("\n".join(ex))

    with st.expander("敏感度测试", expanded=False):
        tweak = st.slider("候选译文截断长度（字符）", min_value=0, max_value=max(0, len(hyp)), value=len(hyp))
        hyp2 = hyp[:tweak]
        d2 = bleu_score_with_details(ref, hyp2, max_n=settings["bleu_max_n"], smooth=settings["bleu_smooth"])
        st.metric("BLEU（截断后）", value=f"{d2.bleu * 100:.2f}")
        st.caption(f"BP={d2.bp:.4f}，候选长度={d2.cand_len}")

    cases = st.session_state.get("bleu_cases", [])
    if cases:
        payload = json.dumps(cases[-20:], ensure_ascii=False, indent=2)
        st.download_button("下载最近 20 条评测案例（JSON）", data=payload.encode("utf-8"), file_name="bleu_cases.json", mime="application/json")


def _tab_help() -> None:
    st.subheader("使用说明")
    st.write("这个应用用于教学演示：体验神经网络翻译、对比规则翻译与神经翻译、用 BLEU 做自动评测。")
    st.write("建议流程：先在“神经网络翻译”里体验，再去“规则 vs 神经网络”观察差异，最后用 BLEU 模块做量化对比。")


def main() -> None:
    st.set_page_config(page_title=DEFAULT_CONFIG.page_title, layout="wide", initial_sidebar_state="expanded")
    st.title(DEFAULT_CONFIG.page_title)

    settings = _render_sidebar()
    examples = _load_examples(BASE_DIR / DEFAULT_CONFIG.examples_dir)

    tabs = st.tabs(["神经网络翻译", "翻译方法对比", "翻译质量评测", "使用说明"])
    with tabs[0]:
        _tab_nmt(settings, examples)
    with tabs[1]:
        _tab_compare(settings, examples)
    with tabs[2]:
        _tab_bleu(settings)
    with tabs[3]:
        _tab_help()

    st.divider()
    st.caption("状态：模型与翻译结果会在会话内缓存；长文本建议使用分块设置降低内存压力。")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        st.error("运行时异常")
        st.code(traceback.format_exc(), language=None)

