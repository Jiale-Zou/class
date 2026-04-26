# 基于 Streamlit 的 NER + 关系抽取 + 知识图谱可视化（教学演示）

这个应用面向 NLP / 信息抽取教学场景，提供“文本 → 实体与关系 → 知识图谱”的可观察链路：

- NER：实体高亮（PERSON / ORG / LOC）
- BIO：一键切换查看底层 BIO 序列
- RE：输出 Subject–Predicate–Object 三元组表
- KG：使用 `vis-network` 在页面内交互式渲染图谱（拖拽、缩放、边标签）

## 运行

```bash
cd ner-re-kg-app
pip install -r requirements.txt
streamlit run app.py
```

## 说明

- 当前默认使用 Mock 数据与规则（便于教学与理解）。
- 图谱渲染通过 `streamlit.components.v1.html` 嵌入 `vis-network` CDN。

