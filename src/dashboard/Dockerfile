FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl libpq-dev gcc && \
    rm -rf /var/lib/apt/lists/*

COPY src/dashboard/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/dashboard/app.py .
COPY src/dashboard/recommendation.py .
COPY src/dashboard/models.py .

EXPOSE 3000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "3000", "--log-level", "info"]
