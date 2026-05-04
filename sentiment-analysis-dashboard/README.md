# 舆情分析工具（Streamlit + Transformers + Plotly）

面向电商/社交媒体场景的轻量级舆情分析原型：支持单文本情感分析、显式/隐式表达对比，以及批量评论的舆情可视化仪表盘展示。

## 项目目录

```
sentiment-analysis-dashboard/
├── app.py                    # 主应用程序入口（多页面导航说明）
├── requirements.txt          # 依赖包列表
├── config.py                 # 配置文件（模型名、阈值、默认参数）
├── utils/                    # 工具函数
│   ├── __init__.py
│   ├── data_generator.py     # 模拟数据生成
│   ├── visualization.py      # 可视化图表函数（Plotly）
│   └── sentiment_analyzer.py # 情感分析封装（Transformers Pipeline）
├── pages/                    # Streamlit 多页面
│   ├── 1_单文本分析.py
│   ├── 2_显式隐式情感.py
│   └── 3_舆情仪表盘.py
└── assets/                   # 静态资源
    └── style.css             # 自定义样式
```

## 项目说明

### 模块1：基础情感分类与置信度量化

- 输入：用户文本
- 输出：情感标签（Positive/Neutral/Negative）+ 置信度（0~1）
- 展示：标签徽章 + Plotly 仪表盘（Gauge）

### 模块2：显式 vs. 隐式情感识别（对比）

- 双栏输入：显式表达（直接情绪词）与隐式表达（事实描述暗含态度）
- 并排展示两种输入的情感标签与置信度，辅助观察模型对不同表达方式的敏感度差异

### 模块3：舆情挖掘与可视化仪表盘

- 生成模拟评论数据（平台、品类、时间、文本）
- 批量情感分析并聚合统计
- 图表组合（Plotly）：
  - 情感分布饼图
  - 平均置信度柱状图（按情感）
  - 高频关键词条形图（作为词云的轻量替代）
  - 评论明细表格（可滚动浏览）

## 快速开始

### 1) 安装依赖

在项目目录下执行：

```bash
pip install -r requirements.txt
```

### 2) 启动应用

```bash
streamlit run app.py
```

启动后，在左侧侧边栏切换页面：
- 单文本分析
- 显式 vs 隐式对比
- 舆情仪表盘

### 3) 可选配置（环境变量）

可通过环境变量覆盖默认配置（见 config.py）：

- HF_MODEL_NAME：情感分类模型名（默认 `uer/roberta-base-finetuned-jd-binary-chinese`）
- NEUTRAL_THRESHOLD：当模型为二分类时，将低置信度结果映射为 Neutral 的阈值（默认 `0.60`）
- DEFAULT_DASHBOARD_ROWS：仪表盘默认模拟评论数量（默认 `60`）

示例（PowerShell）：

```powershell
$env:HF_MODEL_NAME="uer/roberta-base-finetuned-jd-binary-chinese"
$env:NEUTRAL_THRESHOLD="0.60"
streamlit run app.py
```

