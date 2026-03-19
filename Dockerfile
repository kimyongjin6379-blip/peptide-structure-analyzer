# ============================================================
# Peptide Structure Analyzer - Railway Deployment
# PyTorch(CPU) + ESM-2 + Streamlit 통합 컨테이너
# ============================================================

FROM python:3.11-slim

# 시스템 패키지 설치 (HMMER, 빌드 도구 등)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    hmmer \
    clustalo \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리
WORKDIR /app

# ---- 0단계: numpy 1.x 먼저 고정 (PyTorch 2.2 호환) ----
RUN pip install --no-cache-dir "numpy>=1.24.0,<2.0.0"

# ---- 1단계: PyTorch CPU 설치 ----
RUN pip install --no-cache-dir \
    torch==2.2.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# ---- 2단계: ML/DL 핵심 패키지 설치 ----
RUN pip install --no-cache-dir \
    fair-esm==2.0.0 \
    transformers>=4.36.0 \
    tokenizers>=0.15.0 \
    accelerate>=0.25.0 \
    safetensors>=0.4.0

# ---- 3단계: 나머지 requirements 설치 ----
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- 4단계: 앱 코드 복사 ----
COPY . .

# ---- 5단계: 모델 캐시 디렉토리 ----
RUN mkdir -p /app/model_cache /app/data/cache/structures
ENV TORCH_HOME=/app/model_cache
ENV HF_HOME=/app/model_cache
ENV TRANSFORMERS_CACHE=/app/model_cache

# ---- 6단계: Streamlit 설정 ----
RUN mkdir -p /app/.streamlit
COPY .streamlit/config.toml /app/.streamlit/config.toml

# ---- 시작 스크립트 ----
RUN echo '#!/bin/bash\nstreamlit run streamlit_app/app.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true --browser.gatherUsageStats=false' > /app/start.sh \
    && chmod +x /app/start.sh

CMD ["/bin/bash", "/app/start.sh"]
