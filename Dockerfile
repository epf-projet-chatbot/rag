FROM python:3.11-slim

WORKDIR /app

# Copier les requirements et les installer
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copier tout le code RAG
COPY . .

# Ajouter le dossier au PYTHONPATH
ENV PYTHONPATH=/app

# Tu peux modifier la commande selon ce que tu veux faire par défaut
CMD ["python", "chatbot_rag/embedding.py"]