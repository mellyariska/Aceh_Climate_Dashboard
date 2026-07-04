
##############################################################
# modules/xai.py
# Explainable Artificial Intelligence (SHAP)
##############################################################

import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
import plotly.express as px

def shap_dashboard(df):

    st.header("🧠 Explainable Artificial Intelligence (SHAP)")
    st.write("Model Explainability menggunakan SHAP untuk menjelaskan prediksi model XGBoost.")

    model_file = joblib.load("models/xgboost.pkl")
    model = model_file["model"]

    feature_columns = [
        "Rainfall","Tmax","Tmin","Humidity","Pressure",
        "Wind","Solar","ENSO","IOD","NDVI","SPI"
    ]

    X = df[feature_columns]

    try:
        explainer = shap.Explainer(model)
        shap_values = explainer(X)
        use_new = True
    except Exception:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        use_new = False

    st.success("SHAP values successfully calculated.")

    st.markdown("---")
    st.subheader("📊 SHAP Summary Plot")

    fig = plt.figure(figsize=(10,6))
    try:
        if use_new:
            shap.plots.beeswarm(shap_values, show=False)
        else:
            shap.summary_plot(shap_values, X, show=False)
    except Exception:
        try:
            shap.summary_plot(shap_values[0], X, show=False)
        except Exception as e:
            st.error(f"Summary plot gagal: {e}")
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.subheader("📈 SHAP Feature Importance")

    if use_new:
        vals = np.abs(shap_values.values).mean(axis=0)
    else:
        if isinstance(shap_values, list):
            vals = np.mean([np.abs(v).mean(axis=0) for v in shap_values], axis=0)
        else:
            vals = np.abs(shap_values).mean(axis=0)

    importance = pd.DataFrame({
        "Feature": feature_columns,
        "Mean_SHAP": vals
    }).sort_values("Mean_SHAP", ascending=False)

    st.dataframe(importance, use_container_width=True)

    fig2 = px.bar(
        importance,
        x="Mean_SHAP",
        y="Feature",
        orientation="h",
        color="Mean_SHAP"
    )
    st.plotly_chart(fig2, use_container_width=True)

    idx = st.slider("Pilih Observasi",0,len(X)-1,0)

    st.markdown("---")
    st.subheader("🔍 Local Prediction")

    try:
        fig3 = plt.figure(figsize=(10,6))
        if use_new:
            shap.plots.waterfall(shap_values[idx], show=False)
        st.pyplot(fig3)
        plt.close()
    except Exception:
        st.info("Waterfall plot tidak tersedia untuk konfigurasi model ini.")

    csv = importance.to_csv(index=False)
    st.download_button(
        "📥 Download SHAP Importance",
        csv,
        file_name="shap_importance.csv",
        mime="text/csv"
    )
