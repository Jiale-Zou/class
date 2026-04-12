# 篇章分析 Streamlit 应用 — 公网部署用镜像
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 先装 CPU 版 PyTorch，减小体积并避免默认拉 CUDA 轮子
RUN pip install --no-cache-dir "torch>=2.0.0" --index-url https://download.pytorch.org/whl/cpu

COPY requirements-docker.txt requirements-docker.txt
RUN pip install --no-cache-dir -r requirements-docker.txt

COPY . .

RUN python -m spacy download en_core_web_sm

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
