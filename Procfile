web: gunicorn app:app --bind 0.0.0.0:$PORT --timeout 600 --workers 2 --threads 2 --worker-class sync --log-level warning --access-logfile - --error-logfile -
