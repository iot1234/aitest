# Railway/Docker deployment: avoid build-time secrets; use runtime env vars
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY . ./

# Railway provides PORT at runtime. If PORT is missing or invalid (e.g. literal '$PORT'), fall back to 8000.
CMD ["sh", "-c", "PORT_SAFE=\"${PORT:-8000}\"; case \"$PORT_SAFE\" in (''|*[!0-9]*) PORT_SAFE=8000 ;; esac; gunicorn app:app --bind 0.0.0.0:$PORT_SAFE --timeout 600 --workers 1 --threads 4 --worker-class gthread --log-level warning --access-logfile - --error-logfile -"]
