FROM python:3.9

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

COPY data.pdf .

COPY docker-compose.yml .

COPY venv .

CMD ["python", "main.py"]
