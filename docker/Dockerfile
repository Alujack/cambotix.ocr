# Local KhmerOCR REST API.
#
# Model bundles are NOT baked into the image and are never downloaded at
# runtime. Mount an installed models directory read-only:
#
#   docker build -f docker/Dockerfile -t khmerocr-api .
#   docker run --rm -p 127.0.0.1:8000:8000 \
#     -v ~/.khmerocr/models:/models:ro \
#     -e KHMEROCR_MODEL=khmerocr-id-v1 khmerocr-api
#
# Publish the port on 127.0.0.1 only, unless KHMEROCR_API_KEY is set and the
# port sits behind TLS (see docs/api-reference.md, "Access from apps and
# browsers"):
#
#   docker run --rm -p 8000:8000 -v ~/.khmerocr/models:/models:ro \
#     -e KHMEROCR_API_KEY=... -e KHMEROCR_CORS_ORIGINS=https://app.example.com \
#     khmerocr-api

FROM python:3.12-slim AS runtime

# Tesseract engine only; language data comes from the model bundle.
RUN apt-get update \
 && apt-get install -y --no-install-recommends tesseract-ocr \
 && rm -rf /var/lib/apt/lists/* /usr/share/tesseract-ocr/*/tessdata/*.traineddata

WORKDIR /src
COPY packages/core-python /src/packages/core-python
COPY packages/api /src/packages/api
RUN pip install --no-cache-dir "/src/packages/core-python[tesseract,onnx,validation]" /src/packages/api \
 && rm -rf /src

RUN useradd --system --uid 10001 --home-dir /nonexistent khmerocr \
 && mkdir -p /models && chown khmerocr /models
USER khmerocr

# KHMEROCR_HOME/models is where bundles are resolved by ID.
ENV KHMEROCR_HOME=/ \
    KHMEROCR_MODEL=khmerocr-id-v1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TMPDIR=/tmp

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/v1/health').status==200 else 1)"

# Inside the container the server must listen on all interfaces so the
# published port works; restrict exposure with the -p binding instead.
CMD ["khmerocr", "serve", "--host", "0.0.0.0", "--port", "8000"]
