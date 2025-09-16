FROM python:3.11-slim

WORKDIR /app

COPY requirements_api.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements_api.txt

COPY app.py .

EXPOSE 8000

CMD ["python", "app.py"] 