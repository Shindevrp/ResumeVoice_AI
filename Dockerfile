FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake pkg-config libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir ".[all]"


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
ENV RESUMEVOICE_STT_DEVICE=cpu
ENV RESUMEVOICE_STT_COMPUTE=int8
ENV RESUMEVOICE_LLM_URL=http://vllm:8000/v1
ENV RESUMEVOICE_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct-AWQ
ENV RESUMEVOICE_TTS_MODEL=/app/models/piper/en_US-lessac-medium.onnx
ENV RESUMEVOICE_VAD_THRESHOLD=0.5

RUN mkdir -p /app/models/piper && python3 <<EOF
import urllib.request, pathlib
url = 'https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx'
cfg = url + '.json'
dest = pathlib.Path('/app/models/piper/en_US-lessac-medium.onnx')
if not dest.exists():
    print('Downloading Piper voice model...')
    urllib.request.urlretrieve(url, dest)
    urllib.request.urlretrieve(cfg, dest.with_suffix('.onnx.json'))
    print('Piper model downloaded.')
else:
    print('Piper model already exists.')
EOF

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=5 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]
