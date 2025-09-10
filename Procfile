web: gunicorn app:app --bind 0.0.0.0:$PORT --timeout 600 --workers -1 --threads -1 --worker-class sync --log-level warning --access-logfile - --error-logfile -
