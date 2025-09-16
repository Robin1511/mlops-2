#!/bin/bash

echo "🚀 Démarrage des services MLFlow Model Serving"

# Construire et démarrer les services
echo "📦 Construction des images Docker..."
docker-compose build

echo "🔄 Démarrage des services..."
docker-compose up -d

echo "⏳ Attente du démarrage des services (60s)..."
sleep 60

echo "🧪 Exécution des tests..."
python test_api.py

echo "📊 Vérification des logs..."
docker-compose logs model-api

echo "✅ Tests terminés ! Services disponibles sur:"
echo "   - MLFlow UI: http://localhost:5000"
echo "   - Model API: http://localhost:8000"
echo "   - API Docs: http://localhost:8000/docs"

echo "⏹️  Pour arrêter les services: docker-compose down" 