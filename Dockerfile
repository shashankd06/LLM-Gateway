FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /service
COPY pyproject.toml ./
COPY requirements.txt ./
COPY app ./app
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir --no-deps . \
    && useradd --create-home gateway
COPY alembic.ini ./
USER gateway
EXPOSE 8000
CMD ["sh", "-c", "alembic upgrade head && exec uvicorn app.main:create_app --factory --host 0.0.0.0 --port 8000"]
