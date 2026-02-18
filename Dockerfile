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

# Railway provides PORT at runtime
CMD ["sh", "-c", "gunicorn app:app --bind 0.0.0.0:${PORT:-8000} --timeout 600 --workers 1 --threads 4 --worker-class gthread --log-level warning --access-logfile - --error-logfile -"]
