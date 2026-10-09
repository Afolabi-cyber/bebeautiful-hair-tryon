# ---- 1. Build the React app ----
FROM node:22-slim AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ---- 2. Python server ----
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
# System libraries the native code links against:
#   MediaPipe 1.x: libEGL.so.1, libGLESv2.so.2
#   OpenCV (pulled in by MediaPipe): libGL, glib/gthread, X11/xcb client libs
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      libegl1 libgles2 libgl1 libglib2.0-0 \
      libx11-6 libxext6 libsm6 libice6 libxcb1 \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY server/requirements.txt server/requirements.txt
RUN pip install -r server/requirements.txt
COPY server/ server/
COPY wigs/ wigs/
COPY catalog.json ./
COPY --from=web /web/dist frontend/dist
# Bake both AI models into the image so nothing downloads while customers wait,
# then load them once: if a system library is missing, the BUILD fails here
# (with a clear error) instead of the live server failing its health checks.
RUN cd server && python download_models.py && python -c "import faces; faces.FaceTool(); faces.Segmenter(); print('models load OK')"
WORKDIR /app/server
EXPOSE 10000
# One worker: the models, rate limits and caches live in this process.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1 --proxy-headers --forwarded-allow-ips='*' --timeout-keep-alive 75"]
