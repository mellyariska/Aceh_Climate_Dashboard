##############################################################
# modules/report.py
# Climate Intelligence Executive Report
##############################################################

import os
import joblib
import numpy as np
import pandas as pd

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from datetime import datetime


##############################################################
# MAIN FUNCTION
##############################################################

def report_dashboard():

    ##############################################################
    # HEADER
    ##############################################################

    st.header("📑 Climate Intelligence Executive Report")

    st.write("""
Executive dashboard untuk merangkum hasil analisis
Machine Learning, Explainable AI, Decision Support
System dan Early Warning System.
""")

    ##############################################################
    # LOAD MODEL
    ##############################################################

    MODEL_PATH = "models/xgboost.pkl"

    if not os.path.exists(MODEL_PATH):

        st.error("Model XGBoost belum tersedia.")

        st.info(
            "Silakan buka menu XGBoost kemudian lakukan Training Model."
        )

        return

    model_file = joblib.load(MODEL_PATH)

    model = model_file["model"]

    encoder = model_file["encoder"]

    feature_columns = model_file["features"]

    ##############################################################
    # LOAD DATASET
    ##############################################################

    files = [

        "data/data_aceh_1985_2025.xlsx",

        "data_aceh_1985_2025.xlsx",

        "data/data_aceh.xlsx",

        "data_aceh.xlsx"

    ]

    df = None

    for file in files:

        if os.path.exists(file):

            df = pd.read_excel(file)

            break

    if df is None:

        st.error("Dataset tidak ditemukan.")

        return

    ##############################################################
    # CLEAN DATA
    ##############################################################

    df.columns = df.columns.str.strip()

    ##############################################################
    # NUMERIC FEATURE
    ##############################################################

    for col in feature_columns:

        if col in df.columns:

            df[col] = pd.to_numeric(

                df[col],

                errors="coerce"

            )

    df[feature_columns] = df[feature_columns].fillna(

        df[feature_columns].mean()

    )

    ##############################################################
    # TARGET
    ##############################################################

    if "Disaster" in df.columns:

        df["Disaster"] = (

            df["Disaster"]

            .fillna("Normal")

            .astype(str)

            .str.strip()

        )

    st.success(

        f"Dataset berhasil dimuat ({len(df)} observasi)."

    )
        ##############################################################
    # EXECUTIVE SUMMARY
    ##############################################################

    st.markdown("---")

    st.subheader("📋 Executive Summary")

    c1, c2, c3 = st.columns([1,5,1])

    with c1:

        if os.path.exists("assets/logo_unsri.png"):

            st.image(

                "assets/logo_unsri.png",

                width=80

            )

    with c2:

        st.markdown("""

# 🌦 Climate Intelligence Platform

### Machine Learning • Explainable AI • Decision Support System

Aceh Extreme Climate Dashboard

""")

    with c3:

        if os.path.exists("assets/logo_bmkg.png"):

            st.image(

                "assets/logo_bmkg.png",

                width=80

            )

    ##############################################################
    # INFORMATION
    ##############################################################

    today = datetime.now()

    info1, info2 = st.columns(2)

    with info1:

        st.info(f"""

### Report Date

{today.strftime("%d %B %Y")}

""")

    with info2:

        st.info(f"""

### Machine Learning

Model : XGBoost

Explainability : SHAP

Prediction Classes : {len(encoder.classes_)}

""")

    ##############################################################
    # KPI
    ##############################################################

    st.markdown("---")

    k1,k2,k3,k4 = st.columns(4)

    k1.metric(

        "Observations",

        len(df)

    )

    k2.metric(

        "Variables",

        len(df.columns)

    )

    k3.metric(

        "Climate Features",

        len(feature_columns)

    )

    k4.metric(

        "Prediction Classes",

        len(encoder.classes_)

    )

    ##############################################################
    # DATASET OVERVIEW
    ##############################################################

    st.markdown("---")

    st.subheader("🌍 Dataset Overview")

    overview = pd.DataFrame({

        "Information":[

            "Observation",

            "Variables",

            "Climate Features",

            "Prediction Classes",

            "Start Year",

            "End Year"

        ],

        "Value":[

            len(df),

            len(df.columns),

            len(feature_columns),

            len(encoder.classes_),

            int(df["Year"].min()),

            int(df["Year"].max())

        ]

    })

    st.dataframe(

        overview,

        use_container_width=True,

        hide_index=True

    )

    ##############################################################
    # PREVIEW
    ##############################################################

    st.markdown("---")

    st.subheader("📊 Dataset Preview")

    st.dataframe(

        df.head(10),

        use_container_width=True,

        hide_index=True

    )

    ##############################################################
    # DESCRIPTIVE STATISTICS
    ##############################################################

    st.markdown("---")

    st.subheader("📈 Climate Statistics")

    st.dataframe(

        df[feature_columns].describe().round(2),

        use_container_width=True

    )
        ##############################################################
    # MACHINE LEARNING PERFORMANCE
    ##############################################################

    st.markdown("---")

    st.subheader("🎯 Machine Learning Performance")

    ##############################################################
    # FEATURE IMPORTANCE SCORE
    ##############################################################

    importance = pd.DataFrame({

        "Feature": feature_columns,

        "Importance": model.feature_importances_

    })

    importance = importance.sort_values(

        "Importance",

        ascending=False

    )

    ##############################################################
    # MODEL INFORMATION
    ##############################################################

    k1,k2,k3,k4 = st.columns(4)

    k1.metric(

        "Algorithm",

        "XGBoost"

    )

    k2.metric(

        "Features",

        len(feature_columns)

    )

    k3.metric(

        "Classes",

        len(encoder.classes_)

    )

    k4.metric(

        "Top Feature",

        importance.iloc[0]["Feature"]

    )

    ##############################################################
    # FEATURE IMPORTANCE
    ##############################################################

    fig = px.bar(

        importance,

        x="Importance",

        y="Feature",

        orientation="h",

        color="Importance",

        color_continuous_scale="Viridis",

        text_auto=".4f"

    )

    fig.update_layout(

        height=550,

        title="Model Feature Importance"

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    ##############################################################
    # MODEL SUMMARY
    ##############################################################

    st.success(f"""

### Model Summary

Algorithm : **Extreme Gradient Boosting**

Climate Features : **{len(feature_columns)}**

Prediction Classes : **{len(encoder.classes_)}**

Most Important Variable :

**{importance.iloc[0]['Feature']}**

Dashboard menggunakan model Machine Learning
yang telah dilatih sebelumnya untuk melakukan
prediksi kondisi iklim.

""")
        ##############################################################
    # PREDICTION SUMMARY
    ##############################################################

    st.markdown("---")
    st.subheader("🔮 Prediction Summary")

    ##############################################################
    # LAST OBSERVATION
    ##############################################################

    latest = df[feature_columns].tail(1)

    prediction = model.predict(latest)[0]

    probability = model.predict_proba(latest)[0]

    predicted_class = encoder.inverse_transform([prediction])[0]

    confidence = float(np.max(probability) * 100)

    ##############################################################
    # RISK LEVEL
    ##############################################################

    if confidence < 40:

        risk_level = "LOW"

        color = "🟢"

    elif confidence < 70:

        risk_level = "MODERATE"

        color = "🟡"

    else:

        risk_level = "HIGH"

        color = "🔴"

    ##############################################################
    # KPI
    ##############################################################

    c1,c2,c3 = st.columns(3)

    c1.metric(

        "Prediction",

        predicted_class

    )

    c2.metric(

        "Confidence",

        f"{confidence:.2f}%"

    )

    c3.metric(

        "Risk Level",

        risk_level

    )

    ##############################################################
    # PROBABILITY TABLE
    ##############################################################

    prob_df = pd.DataFrame({

        "Class":encoder.classes_,

        "Probability (%)":np.round(

            probability*100,

            2

        )

    })

    st.dataframe(

        prob_df,

        use_container_width=True,

        hide_index=True

    )

    ##############################################################
    # BAR CHART
    ##############################################################

    fig = px.bar(

        prob_df,

        x="Class",

        y="Probability (%)",

        color="Probability (%)",

        text="Probability (%)",

        color_continuous_scale="Turbo"

    )

    fig.update_layout(

        title="Prediction Probability",

        height=500

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )
        ##############################################################
    # FEATURE IMPORTANCE
    ##############################################################

    st.markdown("---")

    st.subheader("🌳 Climate Variable Importance")

    ##############################################################
    # IMPORTANCE TABLE
    ##############################################################

    importance = pd.DataFrame({

        "Feature":feature_columns,

        "Importance":model.feature_importances_

    })

    importance = importance.sort_values(

        "Importance",

        ascending=False

    ).reset_index(drop=True)

    ##############################################################
    # KPI
    ##############################################################

    c1,c2,c3 = st.columns(3)

    c1.metric(

        "Top Feature",

        importance.iloc[0]["Feature"]

    )

    c2.metric(

        "Importance",

        f"{importance.iloc[0]['Importance']:.4f}"

    )

    c3.metric(

        "Variables",

        len(feature_columns)

    )

    ##############################################################
    # TABLE
    ##############################################################

    st.dataframe(

        importance,

        use_container_width=True,

        hide_index=True

    )

    ##############################################################
    # BAR
    ##############################################################

    fig = px.bar(

        importance,

        x="Importance",

        y="Feature",

        orientation="h",

        color="Importance",

        color_continuous_scale="Viridis",

        text_auto=".4f"

    )

    fig.update_layout(

        height=600,

        title="Feature Importance"

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    ##############################################################
    # PIE
    ##############################################################

    fig2 = px.pie(

        importance.head(10),

        names="Feature",

        values="Importance",

        hole=0.45,

        color_discrete_sequence=px.colors.qualitative.Set3

    )

    fig2.update_layout(

        title="Top 10 Variables",

        height=500

    )

    st.plotly_chart(

        fig2,

        use_container_width=True

    )

    st.success(f"""

Most influential climate variable:

### {importance.iloc[0]['Feature']}

Importance Score:

### {importance.iloc[0]['Importance']:.4f}

""")
        ##############################################################
    # EARLY WARNING SYSTEM
    ##############################################################

    st.markdown("---")

    st.subheader("🚨 Early Warning System")

    ##############################################################
    # CLIMATE RISK INDEX
    ##############################################################

    climate_risk = confidence

    if climate_risk < 40:

        status = "LOW"

        message = "🟢 Normal Condition"

    elif climate_risk < 70:

        status = "MODERATE"

        message = "🟡 Climate Watch"

    else:

        status = "HIGH"

        message = "🔴 Extreme Climate Warning"

    ##############################################################
    # KPI
    ##############################################################

    c1,c2,c3 = st.columns(3)

    c1.metric(

        "Climate Risk Index",

        f"{climate_risk:.2f}%"

    )

    c2.metric(

        "Risk Status",

        status

    )

    c3.metric(

        "Warning",

        message

    )

    ##############################################################
    # GAUGE
    ##############################################################

    gauge = go.Figure(

        go.Indicator(

            mode="gauge+number",

            value=climate_risk,

            title={"text":"Climate Risk Index (%)"},

            gauge={

                "axis":{"range":[0,100]},

                "bar":{"color":"royalblue"},

                "steps":[

                    {"range":[0,40],"color":"#b6d7a8"},

                    {"range":[40,70],"color":"#ffe599"},

                    {"range":[70,100],"color":"#f4cccc"}

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
    # ALERT
    ##############################################################

    if status == "LOW":

        st.success("""

### 🟢 LOW RISK

Climate condition is stable.

Routine monitoring is sufficient.

""")

    elif status == "MODERATE":

        st.warning("""

### 🟡 MODERATE RISK

Potential climate anomaly detected.

Increase preparedness and monitoring.

""")

    else:

        st.error("""

### 🔴 HIGH RISK

High probability of extreme climate event.

Early Warning System should be activated immediately.

""")
            ##############################################################
    # DECISION SUPPORT RECOMMENDATION
    ##############################################################

    st.markdown("---")

    st.subheader("🧭 Decision Support Recommendation")

    ##############################################################
    # RECOMMENDATION
    ##############################################################

    if risk_level == "LOW":

        recommendation = [

            "Continue routine climate monitoring.",

            "Maintain existing environmental management.",

            "Update climate database periodically.",

            "Perform regular validation of climate sensors."

        ]

    elif risk_level == "MODERATE":

        recommendation = [

            "Increase climate monitoring frequency.",

            "Prepare local disaster response teams.",

            "Coordinate with BMKG and BPBD.",

            "Disseminate early information to stakeholders."

        ]

    else:

        recommendation = [

            "Activate the Early Warning System immediately.",

            "Coordinate emergency response with local government.",

            "Issue public warnings through official channels.",

            "Prepare evacuation plans for vulnerable communities."

        ]

    for i, rec in enumerate(recommendation, start=1):

        st.write(f"**{i}.** {rec}")

    ##############################################################
    # DECISION SUPPORT SCORE
    ##############################################################

    st.markdown("---")

    st.subheader("📊 Decision Support Score")

    score = confidence

    fig = go.Figure(

        go.Indicator(

            mode="gauge+number",

            value=score,

            title={"text":"Decision Support Score (%)"},

            gauge={

                "axis":{"range":[0,100]},

                "bar":{"color":"royalblue"},

                "steps":[

                    {"range":[0,40],"color":"#b6d7a8"},

                    {"range":[40,70],"color":"#ffe599"},

                    {"range":[70,100],"color":"#f4cccc"}

                ]

            }

        )

    )

    fig.update_layout(

        height=350

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    ##############################################################
    # EXECUTIVE CONCLUSION
    ##############################################################

    st.markdown("---")

    st.success(f"""

### Executive Conclusion

**Predicted Climate Condition**

: **{predicted_class}**

**Prediction Confidence**

: **{confidence:.2f}%**

**Climate Risk Level**

: **{risk_level}**

**Most Influential Climate Variable**

: **{importance.iloc[0]['Feature']}**

The Climate Intelligence Platform recommends that
decision makers use these prediction results to support
climate adaptation, disaster mitigation, and regional
planning.

""")
        ##############################################################
    # EXPORT REPORT
    ##############################################################

    st.markdown("---")

    st.subheader("📥 Export Executive Report")

    ##############################################################
    # SUMMARY TABLE
    ##############################################################

    summary = pd.DataFrame({

        "Parameter":[

            "Prediction",

            "Confidence (%)",

            "Risk Level",

            "Top Feature",

            "Importance",

            "Climate Variables",

            "Prediction Classes",

            "Observation"

        ],

        "Value":[

            predicted_class,

            round(confidence,2),

            risk_level,

            importance.iloc[0]["Feature"],

            round(float(importance.iloc[0]["Importance"]),4),

            len(feature_columns),

            len(encoder.classes_),

            len(df)

        ]

    })

    st.dataframe(

        summary,

        use_container_width=True,

        hide_index=True

    )

    ##############################################################
    # DOWNLOAD CSV
    ##############################################################

    csv = summary.to_csv(index=False)

    st.download_button(

        "📥 Download Executive Report (CSV)",

        data=csv,

        file_name="Executive_Report.csv",

        mime="text/csv"

    )

    ##############################################################
    # DOWNLOAD FEATURE IMPORTANCE
    ##############################################################

    csv2 = importance.to_csv(index=False)

    st.download_button(

        "📥 Download Feature Importance",

        data=csv2,

        file_name="Feature_Importance.csv",

        mime="text/csv"

    )

    ##############################################################
    # FOOTER
    ##############################################################

    st.markdown("---")

    st.markdown("""

<div style="text-align:center;padding:25px">

<h3>🌦 Climate Intelligence Platform</h3>

<b>Machine Learning • Explainable AI • Decision Support System</b>

<br><br>

Developed by

<b>Melly Ariska</b>

<br>

Physics Education Department

Faculty of Teacher Training and Education

Universitas Sriwijaya

<br><br>

Aceh Extreme Climate Dashboard

Version 2.0 | 2026

</div>

""",

    unsafe_allow_html=True)
