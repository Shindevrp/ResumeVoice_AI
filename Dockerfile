FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake pkg-config libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY pyproject.toml ./

COPY wheels/ /wheels/
ENV PIP_DEFAULT_TIMEOUT=120
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --resume-retries 10 --retries 10 --find-links=/wheels ".[all]" && \
    rm -rf /wheels


FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 espeak-ng \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

WORKDIR /app

COPY app/ app/
COPY core/ core/
COPY modules/ modules/
COPY providers/ providers/
COPY utils/ utils/
COPY resume/ resume/

ENV RESUMEVOICE_STT_MODEL=base
ENV RESUMEVOICE_STT_DEVICE=cuda
ENV RESUMEVOICE_STT_COMPUTE=float16
ENV RESUMEVOICE_LLM_URL=http://vllm:8000/v1
ENV RESUMEVOICE_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct-AWQ
ENV RESUMEVOICE_TTS_MODEL=/app/models/piper/en_US-lessac-medium.onnx
ENV RESUMEVOICE_TTS_DEVICE=cuda
ENV RESUMEVOICE_VAD_DEVICE=cuda
ENV RESUMEVOICE_VAD_THRESHOLD=0.45

RUN mkdir -p /app/models/piper

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=5 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]
