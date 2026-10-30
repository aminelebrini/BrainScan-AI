# BrainScan-AI

## Comparer le feature extraction et le fine-tuning

Les historiques Keras présents dans `results/feature_extraction/` et
`results/fin-tuining/` peuvent être comparés avec Matplotlib et Seaborn :

```bash
python main.py
```

Après l'entraînement, `main.py` instancie `Visualisation` et appelle
`display_training_results()`. La méthode affiche les courbes et crée :

- `training_comparison.png` : courbes d'accuracy et de loss pour l'entraînement
  et la validation ;

## Interface Streamlit

L'interface charge les modèles sauvegardés sans relancer leur entraînement :

```bash
streamlit run streamlit_app.py
```

Elle permet de charger une image IRM, d'afficher la classe et les probabilités
du classifieur, les boîtes YOLO avec leur confiance et les métriques sauvegardées
dans `results/visualizations/evaluation_metrics.json`.

## Docker

Le service Streamlit et l'infrastructure Airflow sont définis dans
`docker-compose.yaml`. Après avoir installé Docker, démarrez l'interface avec :

```bash
docker compose up --build streamlit
```

L'interface est disponible sur <http://localhost:8501>. Pour démarrer également
Airflow et ses dépendances :

```bash
docker compose up -d
```

Les modèles et résultats sont montés en lecture seule dans le conteneur
Streamlit afin que l'application ne puisse pas les modifier.

Pour Airflow, configurez une clé Fernet dans `.env` avant un déploiement
persistant (elle peut rester vide pour un usage local sans secrets chiffrés) :

```bash
echo "FERNET_KEY=$(python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())')" >> .env
```

## Tests

Les tests de prétraitement, de format des probabilités et de lecture des
métriques peuvent être exécutés sans charger les modèles lourds :

```bash
python -m unittest discover -s tests -v
```
