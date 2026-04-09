# Utiliser une image Python légère
FROM python:3.12-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers nécessaires
COPY requirements.txt ./
COPY data/ ./data/
COPY data/templates/ ./data/templates/
COPY train.py ./

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Entraîner le modèle (si nécessaire)
RUN python train.py

# Exposer le port 5000
EXPOSE 5000

# Lancer l'application
CMD ["python", "data/app.py"]