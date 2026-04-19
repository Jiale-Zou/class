import math
import os
import time

import pandas as pd
import streamlit as st

from utils.evaluation import (
    compute_gpt2_perplexities,
    detect_repeated_ngrams,
    safe_device,
)
from utils.ngram_utils import (
    build_ngram_model,
    corpus_stats,
    load_nltk_sample_corpus,
    prepare_corpus_text,
    sentence_logprob_details,
)
from utils.pretrained_models import (
    get_bert_fill_mask,
    get_gpt2_generator,
    get_gpt2_lm_and_tokenizer,
)
from utils.rnn_trainer import (
    CharRNNConfig,
    generate_text,
    train_char_model,
)


st.set_page_config(page_title="Language Model Lab", page_icon="🧪", layout="wide")


def inject_global_style() -> None:
    st.markdown(
        """
        <style>
        .stApp {
          background: #0e1117;
        }
        .lm-banner {
          background: linear-gradient(90deg, rgba(31,119,180,0.35), rgba(14,17,23,0));
          border: 1px solid rgba(255,255,255,0.08);
          padding: 18px 18px;
          border-radius: 12px;
          margin-bottom: 14px;
        }
        .lm-title {
          font-size: 34px;
          font-weight: 750;
          line-height: 1.1;
          margin: 0;
        }
        .lm-subtitle {
          color: rgba(255,255,255,0.75);
          margin-top: 6px;
          margin-bottom: 0;
        }
        .lm-kpi {
          border: 1px solid rgba(255,255,255,0.08);
          border-radius: 12px;
          padding: 12px 14px;
          background: rgba(255,255,255,0.02);
        }
        .lm-kpi h3 {
          margin: 0;
          font-size: 13px;
          font-weight: 650;
          color: rgba(255,255,255,0.72);
        }
        .lm-kpi p {
          margin: 6px 0 0 0;
          font-size: 26px;
          font-weight: 800;
          color: #ffffff;
        }
        .lm-chip {
          display: inline-block;
          padding: 3px 10px;
          border-radius: 999px;
          border: 1px solid rgba(255,255,255,0.10);
          background: rgba(255,255,255,0.03);
          font-size: 12px;
          margin-right: 6px;
          margin-top: 6px;
          color: rgba(255,255,255,0.8);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def banner(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="lm-banner">
          <p class="lm-title">{title}</p>
          <p class="lm-subtitle">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="lm-kpi">
          <h3>{label}</h3>
          <p>{value}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


inject_global_style()


tabs = st.tabs(
    [
        "① n-gram 与平滑",
        "② 从零训练 RNN-LM",
        "③ 预训练架构对比",
        "④ 困惑度评估",
    ]
)


with tabs[0]:
    banner(
        "n-gram语言模型与数据平滑",
        "基于语料统计 n-gram 频次，计算句子生成概率，并观察 Laplace（加一）平滑如何缓解零概率问题。",
    )

    with st.expander("理论背景（P11-17、P18-24）", expanded=False):
        st.markdown(
            """
            - n-gram 语言模型：用有限上下文近似条件概率，例如 trigram：P(w_t | w_{t-2}, w_{t-1})
            - 句子联合概率：P(w_1...w_T) = ∏_t P(w_t | context)
            - 零概率问题：训练语料没出现过的 n-gram 导致联合概率为 0
            - Laplace（加一）平滑：count(ngram)+1 / (count(context)+V)，让未见事件获得非零概率
            """
        )

    left, right = st.columns([0.6, 0.4], gap="large")

    with left:
        st.subheader("语料管理")
        corpus_source = st.radio(
            "语料来源选择",
            ["使用NLTK示例语料", "上传文本文件", "手动输入文本"],
            horizontal=True,
            index=0,
        )

        if corpus_source == "使用NLTK示例语料":
            corpus_text = load_nltk_sample_corpus()
            st.caption("来源：NLTK Reuters（取一部分文本用于演示）")
        elif corpus_source == "上传文本文件":
            uploaded = st.file_uploader("上传纯文本文件（.txt）", type=["txt"])
            corpus_text = uploaded.read().decode("utf-8", errors="ignore") if uploaded else ""
        else:
            default_text_path = os.path.join(
                os.path.dirname(__file__), "data", "sample_texts.txt"
            )
            if os.path.exists(default_text_path):
                with open(default_text_path, "r", encoding="utf-8") as f:
                    default_text = f.read()
            else:
                default_text = ""

            corpus_text = st.text_area(
                "输入语料文本",
                value=default_text,
                height=220,
                placeholder="粘贴一段英文语料（建议数百词以上用于观察统计规律）",
            )

        show_preview = st.checkbox("显示语料预览（前200字符）", value=True)
        if show_preview:
            with st.expander("语料内容预览", expanded=True):
                st.code(corpus_text[:200] + ("..." if len(corpus_text) > 200 else ""))

        stats = corpus_stats(corpus_text)
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            kpi("字符数", str(stats["chars"]))
        with s2:
            kpi("单词数", str(stats["words"]))
        with s3:
            kpi("句子数", str(stats["sentences"]))
        with s4:
            kpi("唯一词数", str(stats["vocab"]))

    with right:
        st.subheader("模型配置")
        n_value = st.slider("n-gram大小", min_value=2, max_value=4, value=3, step=1)
        use_laplace = st.toggle("启用加一平滑(Laplace Smoothing)", value=False)
        with st.expander("高级设置", expanded=False):
            unk_mode = st.selectbox(
                "未知词处理",
                [
                    "不处理",
                    "低频替换为<UNK> (freq<2)",
                    "低频替换为<UNK> (freq<3)",
                ],
                index=0,
            )
            verbose_details = st.checkbox("记录详细计算日志", value=True)

    st.divider()

    st.subheader("交互测试")
    test_sentence = st.text_input(
        "测试句子",
        value="The cat sat on the mat",
        placeholder="例如：The cat sat on the mat",
    )

    if "ngram_history" not in st.session_state:
        st.session_state["ngram_history"] = []

    if st.button("计算概率", type="primary"):
        prepared = prepare_corpus_text(corpus_text, unk_mode=unk_mode)
        model = build_ngram_model(prepared, n=n_value)
        details = sentence_logprob_details(
            sentence=test_sentence,
            model=model,
            use_laplace=use_laplace,
        )
        st.session_state["ngram_last"] = details

    details = st.session_state.get("ngram_last")
    if details:
        top_cols = st.columns([0.55, 0.45], gap="large")
        with top_cols[0]:
            prob = math.exp(details["logprob"]) if details["logprob"] > -1e9 else 0.0
            kpi("句子总概率（联合概率）", f"{prob:.4e}")
            chips = []
            if details["had_zero"]:
                chips.append("遇到零概率事件")
            if use_laplace:
                chips.append("使用 Laplace 平滑")
            chips.append(f"n={details['n']}")
            st.markdown(
                "".join([f'<span class="lm-chip">{c}</span>' for c in chips]),
                unsafe_allow_html=True,
            )

        with top_cols[1]:
            if details["had_zero"]:
                st.warning("当前输入句子包含训练语料中未出现的 n-gram，未平滑时联合概率会变为 0。")
            else:
                st.success("当前输入句子的 n-gram 都可在语料中找到（未出现零概率）。")

        with st.expander("每个 n-gram 的计算详情", expanded=verbose_details):
            df = pd.DataFrame(details["rows"])
            st.dataframe(df, use_container_width=True, hide_index=True)

        if st.button("加入对比图表"):
            st.session_state["ngram_history"].append(
                {
                    "sentence": test_sentence,
                    "logprob": details["logprob"],
                    "use_laplace": use_laplace,
                    "n": details["n"],
                }
            )

    if st.session_state["ngram_history"]:
        st.subheader("概率对比图表")
        hist_df = pd.DataFrame(st.session_state["ngram_history"])
        hist_df["prob"] = hist_df["logprob"].apply(
            lambda x: 0.0 if x <= -1e9 else float(math.exp(x))
        )
        st.bar_chart(hist_df.set_index("sentence")[["prob"]])

    with st.expander("观察任务", expanded=False):
        st.markdown(
            """
            - 输入一个“合理但语料未见过”的句子，观察零概率问题是否出现
            - 打开 Laplace 平滑后，比较概率从 0 变为一个很小的非零值
            - 增大 n（例如从 2 到 4）后，零概率问题是否更频繁
            """
        )

        if details:
            st.write(f"当前是否遇到零概率问题：{'是' if details['had_zero'] else '否'}")


with tabs[1]:
    banner(
        "从零训练 RNN 语言模型",
        "使用 PyTorch 训练一个字符级自回归模型：用前 t-1 个字符预测第 t 个字符，并生成连续文本。",
    )

    col_data, col_cfg, col_ctrl = st.columns([0.4, 0.3, 0.3], gap="large")

    with col_data:
        st.subheader("训练数据")
        default_train_text = ("hello world " * 100).strip()
        train_text = st.text_area(
            "输入训练语料（建议数百字符以上）",
            value=st.session_state.get("rnn_train_text", default_train_text),
            height=220,
        )
        st.session_state["rnn_train_text"] = train_text

        preview = train_text[:100] + ("..." if len(train_text) > 100 else "")
        st.caption("数据预览（前100字符）")
        st.code(preview)

        vocab = sorted(set(train_text))
        st.write(f"字符数：{len(train_text)}，词汇量（字符集大小）：{len(vocab)}")
        with st.expander("字符到索引映射表", expanded=False):
            mapping = pd.DataFrame({"char": vocab, "idx": list(range(len(vocab)))})
            st.dataframe(mapping, hide_index=True, use_container_width=True)

    with col_cfg:
        st.subheader("模型配置")
        model_type = st.selectbox("模型类型", ["字符级RNN", "字符级LSTM"], index=1)
        hidden_size = st.slider("隐藏层维度", 16, 128, value=64, step=16)
        num_layers = st.slider("网络层数", 1, 3, value=2, step=1)
        bidirectional = st.toggle("使用双向RNN", value=False)

        st.subheader("训练参数")
        epochs = st.slider("训练轮数", 10, 200, value=50, step=5)
        seq_len = st.slider("序列长度", 20, 100, value=40, step=5)
        optimizer_name = st.selectbox("优化器", ["Adam", "SGD"], index=0)
        lr = st.select_slider(
            "学习率",
            options=[0.001, 0.003, 0.01, 0.03, 0.1],
            value=0.01,
        )
        lr_decay = st.toggle("使用学习率衰减", value=False)

    with col_ctrl:
        st.subheader("训练控制与监控")
        start_train = st.button("▶ 开始训练", type="primary")
        reset_train = st.button("重置训练状态")

        if reset_train:
            for key in ["rnn_model_state", "rnn_history", "rnn_last_metrics"]:
                st.session_state.pop(key, None)

        progress = st.progress(0)
        loss_kpi = st.empty()
        time_kpi = st.empty()
        chart_slot = st.empty()

        if "rnn_history" not in st.session_state:
            st.session_state["rnn_history"] = []

        if start_train:
            cfg = CharRNNConfig(
                model_type="lstm" if "LSTM" in model_type else "rnn",
                hidden_size=hidden_size,
                num_layers=num_layers,
                bidirectional=bidirectional,
                seq_len=seq_len,
                epochs=epochs,
                lr=float(lr),
                optimizer=optimizer_name.lower(),
                lr_decay=lr_decay,
            )
            start_time = time.time()
            history = []
            for step, total_steps, loss_value, model_state, meta in train_char_model(
                train_text=train_text,
                cfg=cfg,
            ):
                history.append({"step": step, "loss": loss_value})
                st.session_state["rnn_model_state"] = model_state
                st.session_state["rnn_last_meta"] = meta

                if total_steps:
                    progress.progress(min(step / total_steps, 1.0))
                loss_kpi.markdown(
                    f"""
                    <div class="lm-kpi"><h3>实时损失（Cross-Entropy）</h3><p>{loss_value:.4f}</p></div>
                    """,
                    unsafe_allow_html=True,
                )
                elapsed = time.time() - start_time
                time_kpi.markdown(
                    f"""
                    <div class="lm-kpi"><h3>训练时间</h3><p>{elapsed:.1f}s</p></div>
                    """,
                    unsafe_allow_html=True,
                )
                chart_slot.line_chart(pd.DataFrame(history).set_index("step"))

            st.session_state["rnn_history"] = history

    st.divider()

    st.subheader("生成与评估")
    seed = st.text_input("起始文本（Seed）", value="hello")
    gen_len = st.slider("生成长度", 10, 200, value=80, step=10)
    temperature = st.slider("温度参数", 0.5, 2.0, value=1.0, step=0.1)
    do_generate = st.button("生成文本")

    model_state = st.session_state.get("rnn_model_state")
    meta = st.session_state.get("rnn_last_meta")

    if model_state and meta and do_generate:
        generated = generate_text(
            model_state=model_state,
            meta=meta,
            seed=seed,
            length=int(gen_len),
            temperature=float(temperature),
        )
        left_out, right_out = st.columns(2, gap="large")
        with left_out:
            st.subheader("原始文本（片段）")
            st.code(st.session_state.get("rnn_train_text", "")[:300])
        with right_out:
            st.subheader("生成文本")
            st.code(generated)

        overlap = 0.0
        if st.session_state.get("rnn_train_text"):
            overlap = float(generated[:50] in st.session_state["rnn_train_text"])
        st.write(f"相似度评分（是否出现训练片段前50字符）：{overlap:.0f}")

        st.download_button(
            "保存生成结果",
            data=generated.encode("utf-8"),
            file_name="rnn_generated.txt",
            mime="text/plain",
        )

    if model_state and meta:
        st.subheader("模型状态信息")
        st.write(f"可训练参数量：{meta['num_parameters']}")
        st.write(f"模型类型：{meta['model_type']}，隐藏维度：{meta['hidden_size']}，层数：{meta['num_layers']}")


with tabs[2]:
    banner(
        "预训练架构对比：Masked LM vs. Causal LM",
        "使用 Hugging Face pipeline 对比 BERT（掩码预测）与 GPT-2（自回归生成）的行为差异。",
    )

    st.markdown(
        """
        | 维度 | BERT（Masked LM） | GPT-2（Causal LM） |
        |---|---|---|
        | 训练目标 | 预测被遮盖的 token | 预测下一个 token |
        | 注意力 | 双向上下文 | 单向（因果掩码） |
        | 典型任务 | 填空、理解 | 续写、生成 |
        """
    )
    st.caption("理论链接：P58-60、P65-71")

    left, right = st.columns(2, gap="large")

    with left:
        st.subheader("BERT（掩码语言模型）")
        examples = [
            "The man went to the [MASK] to buy some milk.",
            "Paris is the capital of [MASK].",
            "The movie was so [MASK] that I watched it twice.",
            "She opened the door with a [MASK].",
            "The cat sat on the [MASK].",
        ]
        example_choice = st.selectbox("示例选择器", examples, index=0)
        mask_sentence = st.text_area(
            "输入带[MASK]的句子",
            value=example_choice,
            height=120,
        )
        top_k = st.slider("Top-K预测数", 3, 10, value=5, step=1)
        show_scores = st.toggle("显示置信度分数", value=True)
        show_attention = st.toggle("显示注意力权重", value=False)

        if st.button("运行 BERT 预测", key="bert_run"):
            fill_mask = get_bert_fill_mask()
            bert_out = fill_mask(mask_sentence, top_k=int(top_k))
            st.session_state["bert_out"] = bert_out

        bert_out = st.session_state.get("bert_out")
        if bert_out:
            st.markdown("**预测候选词（按概率排序）**")
            df = pd.DataFrame(
                [
                    {
                        "token": x["token_str"],
                        "score": float(x["score"]),
                        "sequence": x["sequence"],
                    }
                    for x in bert_out
                ]
            )
            if not show_scores:
                df = df.drop(columns=["score"])
            st.dataframe(df, hide_index=True, use_container_width=True)
            st.bar_chart(pd.DataFrame({"score": [float(x["score"]) for x in bert_out]}, index=[x["token_str"] for x in bert_out]))

            if show_attention:
                with st.expander("注意力热力图（last layer, mean heads）", expanded=True):
                    st.info("为保持演示轻量，这里不默认启用注意力可视化；如需更完整的 token 级可视化，可在 utils/pretrained_models.py 扩展。")

    with right:
        st.subheader("GPT-2（因果语言模型）")
        prompt_examples = [
            "Once upon a time,",
            "In a future where AI helps humans,",
            "The scientist opened the notebook and wrote:",
            "A short recipe for a perfect day:",
            "The meaning of learning is",
        ]
        prompt = st.text_area(
            "输入提示词(Prompt)",
            value=prompt_examples[1],
            height=120,
        )
        if st.button("创意提示（随机）"):
            st.session_state["gpt2_prompt"] = prompt_examples[int(time.time()) % len(prompt_examples)]
            st.rerun()

        prompt = st.session_state.get("gpt2_prompt", prompt)

        gen_words = st.slider("生成长度（词）", 20, 100, value=40, step=5)
        temperature = st.slider("温度", 0.5, 1.5, value=1.0, step=0.1, key="gpt2_temp")
        repetition_penalty = st.slider("重复惩罚", 1.0, 2.0, value=1.1, step=0.05)
        use_beam = st.toggle("使用束搜索", value=False)

        if st.button("运行 GPT-2 生成", key="gpt2_run"):
            generator = get_gpt2_generator()
            gen_out = generator(
                prompt,
                max_new_tokens=int(gen_words) * 2,
                do_sample=not use_beam,
                num_beams=4 if use_beam else 1,
                temperature=float(temperature),
                repetition_penalty=float(repetition_penalty),
                return_full_text=True,
            )
            st.session_state["gpt2_out"] = gen_out[0]["generated_text"]

        gen_text = st.session_state.get("gpt2_out")
        if gen_text:
            st.markdown("**生成结果**")
            show_slot = st.empty()
            prefix_len = len(prompt)
            typed = gen_text[:prefix_len]
            show_slot.code(typed)
            for ch in gen_text[prefix_len:]:
                typed += ch
                show_slot.code(typed)
                time.sleep(0.002)

            st.markdown("**分析面板**")
            device = safe_device(prefer_gpu=False)
            model, tokenizer = get_gpt2_lm_and_tokenizer(device=device)
            ppl = compute_gpt2_perplexities(
                [gen_text],
                model=model,
                tokenizer=tokenizer,
                batch_size=1,
                max_length=256,
                device=device,
            )[0]
            st.write(f"生成文本困惑度（PPL）：{ppl:.2f}")
            rep = detect_repeated_ngrams(gen_text, n=3)
            st.write(f"重复 3-gram 数量：{rep['repeated_ngrams']}")

    st.divider()
    st.subheader("并行对比（相同输入）")
    compare_input = st.text_input("输入一段短文本用于对比", value="The man went to the [MASK].")
    if st.button("运行对比", key="compare_run"):
        fill_mask = get_bert_fill_mask()
        bert_cmp = fill_mask(compare_input, top_k=5)
        generator = get_gpt2_generator()
        gpt_cmp = generator(compare_input.replace("[MASK]", ""), max_new_tokens=40, do_sample=True)[0]["generated_text"]
        st.session_state["cmp"] = {"bert": bert_cmp, "gpt": gpt_cmp}

    cmp_out = st.session_state.get("cmp")
    if cmp_out:
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown("**BERT Top-5**")
            st.dataframe(
                pd.DataFrame(
                    [{"token": x["token_str"], "score": float(x["score"])} for x in cmp_out["bert"]]
                ),
                hide_index=True,
                use_container_width=True,
            )
        with c2:
            st.markdown("**GPT-2 续写**")
            st.code(cmp_out["gpt"])


with tabs[3]:
    banner(
        "语言模型困惑度计算",
        "基于 GPT-2 计算交叉熵并得到 PPL=exp(Loss)。数值越低表示模型越“不困惑”。",
    )

    st.latex(r"\mathrm{PPL} = \exp(\mathrm{cross\_entropy\_loss})")

    left, right = st.columns([0.65, 0.35], gap="large")

    with left:
        st.subheader("测试集管理")
        default_lines = "\n".join(
            [
                "The cat sat on the mat.",
                "Colorless green ideas sleep furiously.",
                "He go to school yesterday.",
                "The quick brown fox jumps over the lazy dog.",
            ]
        )
        lines = st.text_area(
            "输入测试句子（每行一句）",
            value=st.session_state.get("ppl_lines", default_lines),
            height=220,
        )
        st.session_state["ppl_lines"] = lines
        sentences = [s.strip() for s in lines.splitlines() if s.strip()]
        st.caption(f"句子数：{len(sentences)}，字符数：{len(lines)}")

        btn_cols = st.columns(3)
        with btn_cols[0]:
            if st.button("加载示例集"):
                st.session_state["ppl_lines"] = default_lines
                st.rerun()
        with btn_cols[1]:
            if st.button("随机生成句子"):
                st.session_state["ppl_lines"] = "\n".join(
                    [
                        "A small bird sings in the quiet morning.",
                        "The algorithm learns patterns from data.",
                        "This sentence is intentionally strange blue quickly.",
                        "I enjoy reading books about science and art.",
                    ]
                )
                st.rerun()
        with btn_cols[2]:
            if st.button("清除所有"):
                st.session_state["ppl_lines"] = ""
                st.rerun()

    with right:
        st.subheader("评估配置")
        model_choice = st.radio(
            "模型选择",
            ["GPT-2", "BERT（仅演示，PPL不适用）", "自定义模型（高级）"],
            index=0,
        )
        batch_size = st.slider("批次大小", 1, 16, value=4, step=1)
        max_length = st.slider("最大长度（截断）", 32, 512, value=128, step=32)
        prefer_gpu = st.toggle("使用GPU加速（如可用）", value=True)
        standardize_length = st.toggle("标准化长度（按 token 平均）", value=True)

        run_eval = st.button("开始评估", type="primary", disabled=(model_choice != "GPT-2"))

    st.divider()
    st.subheader("评估结果展示")

    if run_eval:
        device = safe_device(prefer_gpu=prefer_gpu)
        model, tokenizer = get_gpt2_lm_and_tokenizer(device=device)
        start = time.time()
        ppls = compute_gpt2_perplexities(
            sentences,
            model=model,
            tokenizer=tokenizer,
            batch_size=int(batch_size),
            max_length=int(max_length),
            device=device,
            standardize_length=standardize_length,
        )
        elapsed = time.time() - start
        result_df = pd.DataFrame({"sentence": sentences, "ppl": ppls})
        result_df["quality"] = pd.cut(
            result_df["ppl"],
            bins=[-1, 50, 100, 200, float("inf")],
            labels=["优秀(<50)", "良好(50-100)", "一般(100-200)", "差(>200)"],
        )
        st.session_state["ppl_df"] = result_df
        st.session_state["ppl_elapsed"] = elapsed

    result_df = st.session_state.get("ppl_df")
    if result_df is not None and not result_df.empty:
        sort_by = st.selectbox("排序", ["ppl 升序", "ppl 降序"], index=0)
        shown = result_df.sort_values("ppl", ascending=(sort_by == "ppl 升序"))
        st.dataframe(shown, hide_index=True, use_container_width=True)

        s1, s2, s3, s4 = st.columns(4)
        with s1:
            kpi("平均PPL", f"{shown['ppl'].mean():.2f}")
        with s2:
            kpi("最小PPL", f"{shown['ppl'].min():.2f}")
        with s3:
            kpi("最大PPL", f"{shown['ppl'].max():.2f}")
        with s4:
            kpi("评估耗时", f"{st.session_state.get('ppl_elapsed', 0.0):.2f}s")

        st.subheader("可视化分析")
        hist = shown["ppl"].clip(upper=500)
        st.bar_chart(pd.DataFrame({"ppl": hist}).reset_index(drop=True))

        lengths = [len(s.split()) for s in shown["sentence"].tolist()]
        scatter_df = pd.DataFrame({"length": lengths, "ppl": shown["ppl"].tolist()})
        st.scatter_chart(scatter_df, x="length", y="ppl")

        csv_bytes = shown.to_csv(index=False).encode("utf-8")
        st.download_button(
            "导出数据(CSV)",
            data=csv_bytes,
            file_name="ppl_results.csv",
            mime="text/csv",
        )
