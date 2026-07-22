##############################################################
# modules/xai.py
# Explainable Artificial Intelligence (SHAP)
# Compatible with XGBoost Multiclass
##############################################################

import os
import joblib
import shap
import numpy as np
import pandas as pd

import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go


##############################################################
# MAIN FUNCTION
##############################################################

def shap_dashboard(df):

    st.header("🧠 Explainable Artificial Intelligence (SHAP)")

    st.markdown("""
Model Explainability menggunakan **SHAP (SHapley Additive Explanations)**

Dashboard ini menjelaskan bagaimana setiap variabel iklim
mempengaruhi hasil prediksi model XGBoost.
""")

    ##############################################################
    # CHECK MODEL
    ##############################################################

    model_path = "models/xgboost.pkl"

    if not os.path.exists(model_path):

        st.error("Model XGBoost belum tersedia.")

        st.info("Silakan buka menu **XGBoost** terlebih dahulu lalu lakukan Training Model.")

        return

   ##############################################################
# LOAD MODEL
##############################################################

MODEL_PATH = "models/xgboost.pkl"

model = None
encoder = None
feature_columns = None

# ==========================================================
# PRIORITAS 1 : Ambil dari Session
# ==========================================================

if "xgb_model" in st.session_state:

    model = st.session_state["xgb_model"]
    encoder = st.session_state["encoder"]
    feature_columns = st.session_state["feature_columns"]

    st.success("✅ XGBoost model loaded from Session.")

# ==========================================================
# PRIORITAS 2 : Ambil dari File
# ==========================================================

elif os.path.exists(MODEL_PATH):

    model_file = joblib.load(MODEL_PATH)

    model = model_file["model"]
    encoder = model_file["encoder"]
    feature_columns = model_file["features"]

    # kalau X_train ikut disimpan
    if "X_train" in model_file:
        X = model_file["X_train"]

    st.success("✅ XGBoost model loaded from File.")

# ==========================================================
# JIKA MODEL BELUM ADA
# ==========================================================

else:

    st.error("❌ XGBoost model belum tersedia.")

    st.info("""
Silakan buka menu **XGBoost** kemudian klik **Train Model** terlebih dahulu.
""")

    st.stop()

st.write("Jumlah Feature :", len(feature_columns))   


# PREPROCESSING
    ##############################################################

    X = df[feature_columns].copy()

    ##############################################################
    # NUMERIC CONVERSION
    ##############################################################

    for col in feature_columns:

        X[col] = pd.to_numeric(

            X[col],

            errors="coerce"

        )

    ##############################################################
    # HANDLE MISSING VALUE
    ##############################################################

    X = X.fillna(

        X.mean(numeric_only=True)

    )

    st.success(f"{len(X)} observasi berhasil dimuat.")

    ##############################################################
    # SHAP EXPLAINER
    ##############################################################

    st.markdown("---")

    st.subheader("⚙ SHAP Explainer")

    with st.spinner("Calculating SHAP values..."):

        explainer = shap.Explainer(model)

        shap_values = explainer(X)

    ##############################################################
    # MULTICLASS COMPATIBILITY
    ##############################################################

    values = shap_values.values

    if values.ndim == 3:

        # Case 1 : (sample, feature, class)

        if values.shape[1] == len(feature_columns):

            shap_matrix = values[:, :, 0]

        # Case 2 : (sample, class, feature)

        elif values.shape[2] == len(feature_columns):

            shap_matrix = values[:, 0, :]

        else:

            st.error("Unknown SHAP output shape.")

            st.write(values.shape)

            return

    else:

        shap_matrix = values

    ##############################################################
    # BUILD SHAP EXPLANATION
    ##############################################################

    base = np.mean(

        np.array(

            shap_values.base_values

        )

    )

    shap_plot = shap.Explanation(

        values=shap_matrix,

        base_values=np.repeat(

            base,

            len(X)

        ),

        data=X.values,

        feature_names=feature_columns

    )

    ##############################################################
    # DEBUG INFORMATION
    ##############################################################

    st.success("SHAP values successfully calculated.")

    st.write("Data Shape :", X.shape)

    st.write("SHAP Shape :", shap_matrix.shape)

    st.write("Feature :", len(feature_columns))
        ##############################################################
    # SHAP SUMMARY PLOT
    ##############################################################

    st.markdown("---")

    st.subheader("📊 SHAP Summary Plot")

    try:

        fig_summary = plt.figure(figsize=(10,6))

        shap.summary_plot(

            shap_plot.values,

            X,

            show=False,

            plot_size=(10,6)

        )

        st.pyplot(fig_summary)

        plt.close()

    except Exception as e:

        st.error(f"Summary plot gagal: {e}")

    ##############################################################
    # SHAP BEESWARM
    ##############################################################

    st.markdown("---")

    st.subheader("🐝 SHAP Beeswarm Plot")

    try:

        fig_bee = plt.figure(figsize=(10,6))

        shap.plots.beeswarm(

            shap_plot,

            max_display=15,

            show=False

        )

        st.pyplot(fig_bee)

        plt.close()

    except Exception as e:

        st.warning(f"Beeswarm plot tidak tersedia : {e}")
            ##############################################################
    # GLOBAL FEATURE IMPORTANCE
    ##############################################################

    st.markdown("---")

    st.subheader("📈 SHAP Global Feature Importance")

    ##############################################################
    # MEAN ABSOLUTE SHAP
    ##############################################################

    importance = pd.DataFrame({

        "Feature": feature_columns,

        "Mean_SHAP": np.abs(shap_matrix).mean(axis=0)

    })

    importance = importance.sort_values(

        by="Mean_SHAP",

        ascending=False

    )

    ##############################################################
    # TABLE
    ##############################################################

    st.dataframe(

        importance,

        use_container_width=True

    )

    ##############################################################
    # BAR CHART
    ##############################################################

    fig_bar = px.bar(

        importance,

        x="Mean_SHAP",

        y="Feature",

        orientation="h",

        color="Mean_SHAP",

        color_continuous_scale="Viridis",

        text_auto=".4f"

    )

    fig_bar.update_layout(

        title="Global SHAP Feature Importance",

        height=600,

        yaxis=dict(categoryorder="total ascending")

    )

    st.plotly_chart(

        fig_bar,

        use_container_width=True

    )

    ##############################################################
    # TOP 10 VARIABLES
    ##############################################################

    st.markdown("---")

    st.subheader("🏆 Top 10 Most Important Features")

    st.dataframe(

        importance.head(10),

        use_container_width=True

    )

    ##############################################################
    # PIE CHART
    ##############################################################

    fig_pie = px.pie(

        importance.head(10),

        names="Feature",

        values="Mean_SHAP",

        hole=0.45,

        color_discrete_sequence=px.colors.qualitative.Set2

    )

    fig_pie.update_layout(

        title="Top Feature Contribution",

        height=500

    )

    st.plotly_chart(

        fig_pie,

        use_container_width=True

    )
        ##############################################################
    # LOCAL EXPLANATION
    ##############################################################

    st.markdown("---")
    st.subheader("🔍 Local Prediction Explanation")

    idx = st.slider(

        "Select Observation",

        min_value=0,

        max_value=len(X)-1,

        value=0

    )

    ##############################################################
    # SELECTED DATA
    ##############################################################

    st.markdown("### Selected Observation")

    st.dataframe(

        X.iloc[[idx]],

        use_container_width=True,

        hide_index=True

    )

    ##############################################################
    # MODEL PREDICTION
    ##############################################################

    pred = model.predict(X.iloc[[idx]])[0]

    prob = model.predict_proba(X.iloc[[idx]])[0]

    label = encoder.inverse_transform([pred])[0]

    st.success(f"Predicted Class : **{label}**")

    ##############################################################
    # PROBABILITY
    ##############################################################

    prob_df = pd.DataFrame({

        "Class": encoder.classes_,

        "Probability": prob

    })

    fig_prob = px.bar(

        prob_df,

        x="Class",

        y="Probability",

        color="Probability",

        color_continuous_scale="Viridis",

        text_auto=".3f"

    )

    fig_prob.update_layout(

        title="Prediction Probability",

        height=450

    )

    st.plotly_chart(

        fig_prob,

        use_container_width=True

    )

    ##############################################################
    # WATERFALL
    ##############################################################

    st.markdown("---")
    st.subheader("💧 SHAP Waterfall Plot")

    try:

        fig_water = plt.figure(figsize=(9,7))

        shap.plots.waterfall(

            shap_plot[idx],

            max_display=15,

            show=False

        )

        st.pyplot(fig_water)

        plt.close()

    except Exception as e:

        st.warning(f"Waterfall Plot tidak tersedia : {e}")
            ##############################################################
    # DEPENDENCE PLOT
    ##############################################################

    st.markdown("---")

    st.subheader("📉 SHAP Dependence Analysis")

    feature = st.selectbox(

        "Select Feature",

        feature_columns,

        index=0

    )

    feature_index = feature_columns.index(feature)

    ##############################################################
    # DEPENDENCE SCATTER
    ##############################################################

    dep_df = pd.DataFrame({

        "Feature Value": X[feature],

        "SHAP Value": shap_matrix[:, feature_index]

    })

    fig_dep = px.scatter(

        dep_df,

        x="Feature Value",

        y="SHAP Value",

        color="SHAP Value",

        color_continuous_scale="Turbo",

        opacity=0.75

    )

    fig_dep.update_layout(

        title=f"Dependence Plot : {feature}",

        height=550

    )

    st.plotly_chart(

        fig_dep,

        use_container_width=True

    )

    ##############################################################
    # FEATURE DISTRIBUTION
    ##############################################################

    st.markdown("---")

    st.subheader("📊 Feature Distribution")

    fig_hist = px.histogram(

        X,

        x=feature,

        nbins=30,

        color_discrete_sequence=["royalblue"]

    )

    fig_hist.update_layout(

        height=420

    )

    st.plotly_chart(

        fig_hist,

        use_container_width=True

    )

    ##############################################################
    # FEATURE STATISTICS
    ##############################################################

    st.markdown("---")

    st.subheader("📋 Feature Statistics")

    stats = pd.DataFrame({

        "Statistic":[

            "Mean",

            "Median",

            "Minimum",

            "Maximum",

            "Std"

        ],

        "Value":[

            X[feature].mean(),

            X[feature].median(),

            X[feature].min(),

            X[feature].max(),

            X[feature].std()

        ]

    })

    st.dataframe(

        stats,

        use_container_width=True,

        hide_index=True

    )

    ##############################################################
    # LOCAL CONTRIBUTION
    ##############################################################

    st.markdown("---")

    st.subheader("🎯 Local Feature Contribution")

    local_df = pd.DataFrame({

        "Feature": feature_columns,

        "Contribution": shap_matrix[idx]

    })

    local_df = local_df.sort_values(

        "Contribution",

        key=np.abs,

        ascending=False

    )

    st.dataframe(

        local_df,

        use_container_width=True,

        hide_index=True

    )

    ##############################################################
    # BAR
    ##############################################################

    fig_local = px.bar(

        local_df,

        x="Contribution",

        y="Feature",

        orientation="h",

        color="Contribution",

        color_continuous_scale="RdBu"

    )

    fig_local.update_layout(

        title="Local SHAP Contribution",

        height=600

    )

    st.plotly_chart(

        fig_local,

        use_container_width=True

    )

    ##############################################################
    # CORRELATION
    ##############################################################

    corr = np.corrcoef(

        X[feature],

        shap_matrix[:, feature_index]

    )[0,1]

    st.info(

        f"Correlation between **{feature}** and SHAP value = **{corr:.3f}**"

    )
        ##############################################################
    # EXPLAINABILITY DASHBOARD
    ##############################################################

    st.markdown("---")

    st.subheader("🎯 Explainability Dashboard")

    ##############################################################
    # EXPLAINABILITY SCORE
    ##############################################################

    explainability_score = np.mean(np.abs(shap_matrix))

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(

        "Observations",

        len(X)

    )

    c2.metric(

        "Features",

        len(feature_columns)

    )

    c3.metric(

        "Top Feature",

        importance.iloc[0]["Feature"]

    )

    c4.metric(

        "Explainability",

        f"{explainability_score:.4f}"

    )

    ##############################################################
    # GAUGE
    ##############################################################

    gauge = go.Figure(

        go.Indicator(

            mode="gauge+number",

            value=explainability_score*100,

            title={"text":"Explainability Score"},

            gauge={

                "axis":{"range":[0,100]},

                "bar":{"color":"royalblue"},

                "steps":[

                    {"range":[0,20],"color":"#ffcccc"},

                    {"range":[20,40],"color":"#ffe599"},

                    {"range":[40,60],"color":"#fff2cc"},

                    {"range":[60,80],"color":"#b6d7a8"},

                    {"range":[80,100],"color":"#6aa84f"}

                ]

            }

        )

    )

    gauge.update_layout(

        height=420

    )

    st.plotly_chart(

        gauge,

        use_container_width=True

    )

    ##############################################################
    # SUMMARY
    ##############################################################

    st.markdown("---")

    st.success(f"""

### Explainable AI Summary

Model berhasil dijelaskan menggunakan SHAP.

Jumlah Observasi :
{len(X)}

Jumlah Variabel :
{len(feature_columns)}

Variabel paling penting :
**{importance.iloc[0]['Feature']}**

Nilai Mean SHAP :
**{importance.iloc[0]['Mean_SHAP']:.5f}**

Explainability Score :
**{explainability_score:.5f}**

Interpretasi :

• Nilai SHAP positif meningkatkan peluang prediksi.

• Nilai SHAP negatif menurunkan peluang prediksi.

• Semakin besar |SHAP| semakin besar kontribusi variabel.

""")
        ##############################################################
    # DOWNLOAD
    ##############################################################

    st.markdown("---")

    st.subheader("📥 Export Explainable AI")

    ##############################################################
    # SHAP IMPORTANCE
    ##############################################################

    csv_importance = importance.to_csv(index=False)

    st.download_button(

        "📥 Download SHAP Importance",

        csv_importance,

        file_name="shap_importance.csv",

        mime="text/csv"

    )

    ##############################################################
    # LOCAL EXPLANATION
    ##############################################################

    csv_local = local_df.to_csv(index=False)

    st.download_button(

        "📥 Download Local Explanation",

        csv_local,

        file_name="local_explanation.csv",

        mime="text/csv"

    )

    ##############################################################
    # COMPLETE REPORT
    ##############################################################

    report = importance.copy()

    report["Explainability_Score"] = explainability_score

    st.download_button(

        "📄 Export Explainable AI Report",

        report.to_csv(index=False),

        file_name="Explainable_AI_Report.csv",

        mime="text/csv"

    )

    ##############################################################
    # FOOTER
    ##############################################################

    st.markdown("---")

    st.caption("""

🧠 Explainable Artificial Intelligence Dashboard

Machine Learning : XGBoost

Explanation Method : SHAP (SHapley Additive Explanations)

Developed by

**Melly Ariska**

Physics Education

Universitas Sriwijaya

2026

Compatible with

✔ XGBoost 2.x

✔ SHAP 0.47+

✔ Streamlit Cloud

""")
