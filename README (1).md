# Prédiction du prix de vente d'une voiture d'occasion

App Streamlit qui prédit le prix de vente (`Selling_Price`, en k$) d'une voiture
d'occasion à partir de 6 variables : kilométrage, prix catalogue neuf, type de
carburant, type de vendeur, transmission et âge du véhicule.

Le modèle retenu est un **Random Forest** (meilleur R² en validation parmi 6
modèles comparés : Linear/Ridge/Lasso Regression, Decision Tree, Random Forest,
XGBoost).

## Fichiers du dépôt

| Fichier | Rôle |
|---|---|
| `app.py` | L'application Streamlit (à déployer) |
| `best_model.joblib` | Le modèle entraîné (Random Forest) |
| `encoders.joblib` | Les `LabelEncoder` des variables catégorielles (Fuel_Type, Seller_Type, Transmission) |
| `model_info.joblib` | Nom du meilleur modèle, ordre des variables, tableau de comparaison des 6 modèles |
| `requirements.txt` | Dépendances pour Streamlit Cloud |
| `train_model.py` | Script pour ré-entraîner et régénérer les 3 fichiers `.joblib` ci-dessus |
| `Car_data.csv` | Données source (utile pour `train_model.py`, pas requis par `app.py`) |

## Déployer sur Streamlit Community Cloud

1. Créer un dépôt GitHub et y pousser au minimum : `app.py`, `best_model.joblib`,
   `encoders.joblib`, `model_info.joblib`, `requirements.txt`.
2. Aller sur [share.streamlit.io](https://share.streamlit.io), se connecter avec GitHub.
3. "New app" → choisir le dépôt, la branche, et `app.py` comme fichier principal.
4. Déployer.

## Ré-entraîner le modèle (optionnel, en local)

```bash
pip install -r requirements-train.txt
python train_model.py   # régénère les .joblib à partir de Car_data.csv
```

## Lancer en local

```bash
pip install -r requirements.txt
streamlit run app.py
```
