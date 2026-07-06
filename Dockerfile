# Pure-Python Warp-Ingest image. No Java, no Apache Tika, no Tesseract binary.
FROM python:3.14-bookworm

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Install from uv.lock with the image's own interpreter — the same frozen
# resolution the CI matrix proves on 3.14 (pip would refuse: rapidocr's
# metadata still says <3.13 even though it runs green on 3.14).
ENV APP_HOME=/root \
  PYTHONUNBUFFERED=1 \
  UV_PYTHON=python3.14 \
  UV_PYTHON_DOWNLOADS=never \
  PATH="/root/.venv/bin:$PATH"

# Minimal system libs.  libgomp1 is needed by onnxruntime (OCR); libmagic1 is an
# optional content-type sniffer used as a fallback by the file property helper.
# (No libGL: the OCR extra uses opencv-python-headless.)
RUN apt-get update && apt-get upgrade -y && \
  apt-get install -y --no-install-recommends \
  libgomp1 \
  libmagic1 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR ${APP_HOME}
COPY . ./

# Install the full service runtime with OCR, then pre-fetch NLTK data and warm
# the bundled OCR models so the service can start fully offline.  Only the model
# warm-up may fail softly; a failed dependency install fails the build.
RUN uv sync --frozen --extra all --no-dev && \
  python -m nltk.downloader -d /usr/share/nltk_data punkt punkt_tab stopwords && \
  { python -c "from rapidocr_onnxruntime import RapidOCR; RapidOCR()" || true; }

RUN chmod +x run.sh

# The FastAPI/uvicorn service budgets its own concurrency at start-up from the
# CPUs actually granted to the container (affinity + cgroup quota), so this
# image deploys at full width on any box without baked-in worker counts.
# Overrides: WARP_WEB_WORKERS, WARP_FE_WORKERS, WARP_OCR_THREADS,
# WARP_WORKER_PARSE_SLOTS, WARP_PORT.  Auth: WARP_API_KEY (default abc123).
EXPOSE 5001
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import os,sys,urllib.request; port=os.environ.get('WARP_PORT') or os.environ.get('PORT') or '5001'; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:%s/healthz' % port, timeout=4).status == 200 else 1)"
CMD ["./run.sh"]
