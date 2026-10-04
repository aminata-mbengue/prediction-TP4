"""
App Streamlit — Prédiction du prix de vente d'une voiture d'occasion

Reprend le pipeline du TP2 : encodage des variables catégorielles (LabelEncoder)
puis Random Forest (meilleur modèle retenu après comparaison).

Fichiers nécessaires dans le même dossier :
- app.py
- best_model.joblib
- encoders.joblib
- model_info.joblib
- requirements.txt
"""

import numpy as np
import pandas as pd
import joblib
import streamlit as st

st.set_page_config(page_title="Prix de voiture d'occasion", page_icon="🚗")


@st.cache_resource
def load_artifacts():
    best_model = joblib.load("best_model.joblib")
    encoders = joblib.load("encoders.joblib")
    info = joblib.load("model_info.joblib")
    return best_model, encoders, info


best_model, encoders, info = load_artifacts()
feature_cols = info["feature_cols"]

st.title("🚗 Prédiction du prix de vente d'une voiture d'occasion")
st.write(
    f"Modèle utilisé : **{info['best_model_name']}** "
    f"(R² validation = {info['results'].iloc[0]['R2']:.3f})"
)


def predict_one(kms_driven, present_price, fuel_type, seller_type, transmission, age):
    fuel_enc = encoders["Fuel_Type"].transform([fuel_type])[0]
    seller_enc = encoders["Seller_Type"].transform([seller_type])[0]
    trans_enc = encoders["Transmission"].transform([transmission])[0]

    x_new = np.array([[kms_driven, present_price, fuel_enc, seller_enc, trans_enc, age]], dtype=float)
    prediction = best_model.predict(x_new)[0]
    return round(float(prediction), 2)


tab1, tab2 = st.tabs(["Prédiction simple", "Prédiction multiple (CSV)"])

with tab1:
    col1, col2 = st.columns(2)
    kms = col1.number_input("Kms parcourus (Kms_Driven)", min_value=0, value=50000, step=1000)
    price = col2.number_input("Prix catalogue neuf en k$ (Present_Price)", min_value=0.0, value=7.5, step=0.1)

    col3, col4, col5 = st.columns(3)
    fuel = col3.selectbox("Type de carburant", list(encoders["Fuel_Type"].classes_))
    seller = col4.selectbox("Type de vendeur", list(encoders["Seller_Type"].classes_))
    trans = col5.selectbox("Transmission", list(encoders["Transmission"].classes_))

    age = st.number_input("Âge du véhicule (années)", min_value=0, value=5, step=1)

    if st.button("Prédire le prix", type="primary"):
        prix = predict_one(kms, price, fuel, seller, trans, age)
        st.success(f"Prix de vente prédit : **{prix} k$**")

with tab2:
    st.write(
        "Importer un fichier CSV avec les colonnes : "
        "`Kms_Driven, Present_Price, Fuel_Type, Seller_Type, Transmission, Age`"
    )
    uploaded_file = st.file_uploader("Fichier CSV", type=["csv"])
    if uploaded_file is not None:
        df_in = pd.read_csv(uploaded_file)
        predictions = [
            predict_one(
                row["Kms_Driven"], row["Present_Price"], row["Fuel_Type"],
                row["Seller_Type"], row["Transmission"], row["Age"],
            )
            for _, row in df_in.iterrows()
        ]
        df_in["Predicted Selling_Price"] = predictions
        st.dataframe(df_in)
        st.download_button(
            "Télécharger les prédictions (CSV)",
            df_in.to_csv(index=False).encode("utf-8"),
            "predictions.csv",
            "text/csv",
        )
