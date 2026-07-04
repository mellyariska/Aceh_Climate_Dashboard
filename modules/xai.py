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

##############################################################
# MAIN FUNCTION
##############################################################

def shap_dashboard(df):

    st.header("🧠 Explainable Artificial Intelligence (SHAP)")

    st.markdown("""
Model Explainability digunakan untuk mengetahui
kontribusi masing-masing variabel iklim terhadap
prediksi Machine Learning.
""")

    ###########################################################
    # LOAD MODEL
    ###########################################################

    model_file = joblib.load("models/xgboost.pkl")

    model = model_file["model"]

    ###########################################################
    # FEATURE
    ###########################################################

    feature_columns = [

        "Rainfall",
        "Tmax",
        "Tmin",
        "Humidity",
        "Pressure",
        "Wind",
        "Solar",
        "ENSO",
        "IOD",
        "NDVI",
        "SPI"

    ]

    X = df[feature_columns]

    ###########################################################
    # SHAP EXPLAINER
    ###########################################################

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X)

    st.success("SHAP values successfully calculated.")

##############################################################
# SHAP SUMMARY PLOT
##############################################################

st.markdown("---")
st.subheader("📊 SHAP Summary Plot")

fig_summary = plt.figure(figsize=(10,6))

try:

    # Untuk SHAP versi baru
    shap.summary_plot(
        shap_values,
        X,
        show=False
    )

except:

    # Fallback untuk multiclass
    shap.summary_plot(
        shap_values[0],
        X,
        show=False
    )

st.pyplot(fig_summary)

plt.close()

##############################################################
# SHAP BAR PLOT
##############################################################

st.markdown("---")
st.subheader("📈 Global Feature Importance (SHAP)")

fig_bar = plt.figure(figsize=(10,6))

try:

    shap.summary_plot(
        shap_values,
        X,
        plot_type="bar",
        show=False
    )

except:

    shap.summary_plot(
        shap_values[0],
        X,
        plot_type="bar",
        show=False
    )

st.pyplot(fig_bar)

plt.close()

##############################################################
# SELECT OBSERVATION
##############################################################

st.markdown("---")

st.subheader("🔍 Local Explanation")

index = st.slider(

    "Select Observation",

    0,

    len(X)-1,

    0

)

##############################################################
# WATERFALL PLOT
##############################################################

st.subheader("💧 SHAP Waterfall Plot")

fig_water = plt.figure(figsize=(10,7))

try:

    shap.plots.waterfall(

        shap.Explanation(

            values=shap_values[index],

            base_values=explainer.expected_value,

            data=X.iloc[index],

            feature_names=X.columns

        ),

        show=False

    )

except:

    pass

st.pyplot(fig_water)

plt.close()

##############################################################
# DEPENDENCE PLOT
##############################################################

st.markdown("---")

st.subheader("📉 SHAP Dependence Plot")

feature = st.selectbox(

    "Feature",

    feature_columns

)

fig_dep = plt.figure(figsize=(8,6))

try:

    shap.dependence_plot(

        feature,

        shap_values,

        X,

        show=False

    )

except:

    pass

st.pyplot(fig_dep)

plt.close()

##############################################################
# FEATURE IMPORTANCE TABLE
##############################################################

st.markdown("---")

st.subheader("Feature Importance (Mean SHAP)")

try:

    importance = np.abs(shap_values).mean(axis=0)

except:

    importance = np.abs(shap_values[0]).mean(axis=0)

importance_df = pd.DataFrame(

    {

        "Feature":feature_columns,

        "Mean_SHAP":importance

    }

)

importance_df = importance_df.sort_values(

    "Mean_SHAP",

    ascending=False

)

st.dataframe(

    importance_df,

    use_container_width=True

)

##############################################################
# PLOTLY FEATURE IMPORTANCE
##############################################################

fig = px.bar(

    importance_df,

    x="Mean_SHAP",

    y="Feature",

    orientation="h",

    color="Mean_SHAP",

    color_continuous_scale="Viridis"

)

fig.update_layout(

    height=550,

    title="SHAP Global Feature Importance"

)

st.plotly_chart(

    fig,

    use_container_width=True

)

##############################################################
# DOWNLOAD SHAP
##############################################################

csv = importance_df.to_csv(index=False)

st.download_button(

    "📥 Download SHAP Importance",

    csv,

    file_name="shap_importance.csv",

    mime="text/csv"

)