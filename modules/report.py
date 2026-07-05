##############################################################
# modules/report.py
# Executive Climate Intelligence Report
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

    st.header("📑 Climate Intelligence Executive Report")

    st.markdown("""
Dashboard ini menyajikan ringkasan hasil analisis
Machine Learning, Explainable Artificial Intelligence,
serta Decision Support System untuk prediksi
bencana iklim.
""")

    ##############################################################
    # LOAD MODEL
    ##############################################################

    model_path = "models/xgboost.pkl"

    if not os.path.exists(model_path):

        st.error("Model XGBoost belum tersedia.")

        st.info(
            "Silakan buka menu **XGBoost** kemudian lakukan Training Model terlebih dahulu."
        )

        return

    model_file = joblib.load(model_path)

    model = model_file["model"]

    encoder = model_file["encoder"]

    feature_columns = model_file["features"]

    ##############################################################
    # LOAD DATASET
    ##############################################################

    data_path = "data/data_aceh_1985_2025.xlsx"

    if os.path.exists(data_path):

        df = pd.read_excel(data_path)

    else:

        st.warning("Dataset tidak ditemukan.")

        df = pd.DataFrame()

    ##############################################################
    # PREPROCESSING
    ##############################################################

    if not df.empty:

        df.columns = df.columns.str.strip()

        for col in feature_columns:

            if col in df.columns:

                df[col] = pd.to_numeric(

                    df[col],

                    errors="coerce"

                )

        df = df.fillna(

            df.mean(

                numeric_only=True

            )

        )

    st.success("✅ Executive Report Loaded Successfully")
    ##############################################################
    # EXECUTIVE SUMMARY
    ##############################################################

    st.markdown("---")

    st.subheader("📋 Executive Summary")

    ##############################################################
    # HEADER
    ##############################################################

    col1, col2, col3 = st.columns([1,4,1])

    with col1:

        if os.path.exists("assets/logo_unsri.png"):

            st.image(

                "assets/logo_unsri.png",

                width=90

            )

    with col2:

        st.markdown("""

# Climate Intelligence Platform

### Machine Learning • Explainable AI • Decision Support System

**Universitas Sriwijaya**

""")

    with col3:

        if os.path.exists("assets/logo_bmkg.png"):

            st.image(

                "assets/logo_bmkg.png",

                width=90

            )

    ##############################################################
    # REPORT INFORMATION
    ##############################################################

    today = datetime.now()

    c1, c2 = st.columns(2)

    with c1:

        st.info(f"""

**Report Date**

{today.strftime("%d %B %Y")}

""")

    with c2:

        st.info("""

**Machine Learning Model**

Extreme Gradient Boosting (XGBoost)

""")

    ##############################################################
    # KPI
    ##############################################################

    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(

        "Observations",

        len(df)

    )

    c2.metric(

        "Climate Features",

        len(feature_columns)

    )

    c3.metric(

        "Prediction Classes",

        len(encoder.classes_)

    )

    c4.metric(

        "Model",

        "XGBoost"

    )

    ##############################################################
    # DATA PREVIEW
    ##############################################################

    st.markdown("---")

    st.subheader("🌦 Climate Dataset Preview")

    if not df.empty:

        st.dataframe(

            df.head(10),

            use_container_width=True,

            hide_index=True

        )

    else:

        st.warning("Dataset belum tersedia.")

    ##############################################################
    # BASIC STATISTICS
    ##############################################################

    st.markdown("---")

    st.subheader("📈 Climate Data Statistics")

    if not df.empty:

        st.dataframe(

            df[feature_columns].describe().round(2),

            use_container_width=True

        )
    ##############################################################
    # MACHINE LEARNING PERFORMANCE
    ##############################################################

    st.markdown("---")

    st.subheader("🎯 Machine Learning Performance")

    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix
    )

    ##############################################################
    # PREPARE DATA
    ##############################################################

    if "Disaster" not in df.columns:

        st.error("Kolom 'Disaster' tidak ditemukan pada dataset.")

        return

    X = df[feature_columns].copy()

    y = df["Disaster"].astype(str)

    ##############################################################
    # HANDLE MISSING VALUE
    ##############################################################

    X = X.apply(pd.to_numeric, errors="coerce")

    X = X.fillna(X.mean())

    ##############################################################
    # ENCODE TARGET
    ##############################################################

    y_true = encoder.transform(y)

    ##############################################################
    # PREDICTION
    ##############################################################

    y_pred = model.predict(X)

    ##############################################################
    # METRICS
    ##############################################################

    acc = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    ##############################################################
    # KPI
    ##############################################################

    c1,c2,c3,c4 = st.columns(4)

    c1.metric("Accuracy",f"{acc:.3f}")

    c2.metric("Precision",f"{precision:.3f}")

    c3.metric("Recall",f"{recall:.3f}")

    c4.metric("F1 Score",f"{f1:.3f}")

    ##############################################################
    # GAUGE
    ##############################################################

    gauge = go.Figure(

        go.Indicator(

            mode="gauge+number",

            value=acc*100,

            title={"text":"Overall Accuracy (%)"},

            gauge={

                "axis":{"range":[0,100]},

                "bar":{"color":"royalblue"},

                "steps":[

                    {"range":[0,60],"color":"#ffb3b3"},

                    {"range":[60,80],"color":"#ffe599"},

                    {"range":[80,90],"color":"#b6d7a8"},

                    {"range":[90,100],"color":"#6aa84f"}

                ]

            }

        )

    )

    gauge.update_layout(height=420)

    st.plotly_chart(
        gauge,
        use_container_width=True
    )

    ##############################################################
    # BAR PERFORMANCE
    ##############################################################

    perf = pd.DataFrame({

        "Metric":[
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score"
        ],

        "Value":[
            acc,
            precision,
            recall,
            f1
        ]

    })

    fig_perf = px.bar(

        perf,

        x="Metric",

        y="Value",

        color="Value",

        color_continuous_scale="Viridis",

        text_auto=".3f"

    )

    fig_perf.update_layout(

        height=450

    )

    st.plotly_chart(

        fig_perf,

        use_container_width=True

    )
    ##############################################################
    # PREDICTION SUMMARY
    ##############################################################

    st.markdown("---")

    st.subheader("🌦 Prediction Summary")

    ##############################################################
    # LAST OBSERVATION
    ##############################################################

    latest = X.tail(1)

    pred = model.predict(latest)[0]

    prob = model.predict_proba(latest)[0]

    label = encoder.inverse_transform([pred])[0]

    risk = float(np.max(prob)*100)

    ##############################################################
    # RISK LEVEL
    ##############################################################

    if risk < 40:

        level = "LOW"

    elif risk < 70:

        level = "MODERATE"

    else:

        level = "HIGH"

    ##############################################################
    # KPI
    ##############################################################

    c1,c2,c3 = st.columns(3)

    c1.metric(

        "Predicted Disaster",

        label

    )

    c2.metric(

        "Confidence",

        f"{risk:.2f}%"

    )

    c3.metric(

        "Risk Level",

        level

    )

    ##############################################################
    # PROBABILITY TABLE
    ##############################################################

    prob_df = pd.DataFrame({

        "Disaster":encoder.classes_,

        "Probability":prob

    })

    prob_df["Probability (%)"] = (

        prob_df["Probability"]*100

    ).round(2)

    st.dataframe(

        prob_df,

        use_container_width=True,

        hide_index=True

    )

    ##############################################################
    # BAR CHART
    ##############################################################

    fig_prob = px.bar(

        prob_df,

        x="Disaster",

        y="Probability (%)",

        color="Probability (%)",

        text="Probability (%)",

        color_continuous_scale="Turbo"

    )

    fig_prob.update_layout(

        height=500,

        title="Prediction Probability"

    )

    st.plotly_chart(

        fig_prob,

        use_container_width=True

    )

    ##############################################################
    # PIE CHART
    ##############################################################

    fig_pie = px.pie(

        prob_df,

        names="Disaster",

        values="Probability (%)",

        hole=0.45,

        color_discrete_sequence=px.colors.qualitative.Set2

    )

    fig_pie.update_layout(

        height=450

    )

    st.plotly_chart(

        fig_pie,

        use_container_width=True
    )
    ##############################################################
    # FEATURE IMPORTANCE
    ##############################################################

    st.markdown("---")

    st.subheader("🧠 Explainable AI & Feature Importance")

    ##############################################################
    # XGBOOST FEATURE IMPORTANCE
    ##############################################################

    importance = pd.DataFrame({

        "Feature": feature_columns,

        "Importance": model.feature_importances_

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

        "Total Features",

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
    # BAR CHART
    ##############################################################

    fig_bar = px.bar(

        importance,

        x="Importance",

        y="Feature",

        orientation="h",

        color="Importance",

        text_auto=".4f",

        color_continuous_scale="Viridis"

    )

    fig_bar.update_layout(

        height=600,

        title="XGBoost Feature Importance"

    )

    st.plotly_chart(

        fig_bar,

        use_container_width=True

    )

    ##############################################################
    # PIE CHART
    ##############################################################

    fig_pie = px.pie(

        importance.head(10),

        names="Feature",

        values="Importance",

        hole=0.45,

        color_discrete_sequence=px.colors.qualitative.Set3

    )

    fig_pie.update_layout(

        height=500,

        title="Top 10 Feature Contribution"

    )

    st.plotly_chart(

        fig_pie,

        use_container_width=True

    )

    ##############################################################
    # INTERPRETATION
    ##############################################################

    st.success(f"""

Top predictor berdasarkan model XGBoost adalah

**{importance.iloc[0]['Feature']}**

dengan nilai importance sebesar

**{importance.iloc[0]['Importance']:.4f}**

Variabel tersebut memiliki kontribusi terbesar
terhadap prediksi bencana iklim.

""")
    ##############################################################
    # EARLY WARNING SYSTEM
    ##############################################################

    st.markdown("---")

    st.subheader("🚨 Early Warning System")

    ##############################################################
    # CLIMATE RISK INDEX
    ##############################################################

    climate_risk = risk

    if climate_risk < 40:

        status = "LOW"

        alert = "🟢 NORMAL"

        color = "green"

    elif climate_risk < 70:

        status = "MODERATE"

        alert = "🟡 WATCH"

        color = "orange"

    else:

        status = "HIGH"

        alert = "🔴 WARNING"

        color = "red"

    ##############################################################
    # KPI
    ##############################################################

    c1,c2,c3 = st.columns(3)

    c1.metric(

        "Climate Risk Index",

        f"{climate_risk:.1f}%"

    )

    c2.metric(

        "Risk Status",

        status

    )

    c3.metric(

        "Early Warning",

        alert

    )

    ##############################################################
    # GAUGE
    ##############################################################

    fig = go.Figure(

        go.Indicator(

            mode="gauge+number",

            value=climate_risk,

            title={"text":"Climate Risk Index"},

            gauge={

                "axis":{"range":[0,100]},

                "bar":{"color":"royalblue"},

                "steps":[

                    {

                        "range":[0,40],

                        "color":"#b6d7a8"

                    },

                    {

                        "range":[40,70],

                        "color":"#ffe599"

                    },

                    {

                        "range":[70,100],

                        "color":"#f4cccc"

                    }

                ]

            }

        )

    )

    fig.update_layout(

        height=450

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    ##############################################################
    # ALERT BOX
    ##############################################################

    if status == "LOW":

        st.success("""

### 🟢 Climate Condition

Climate conditions are stable.

Routine monitoring is recommended.

""")

    elif status == "MODERATE":

        st.warning("""

### 🟡 Climate Watch

Potential climate anomaly detected.

Increase monitoring activities.

""")

    else:

        st.error("""

### 🔴 Climate Warning

High probability of extreme climate event.

Early Warning System should be activated immediately.

""")

    ##############################################################
    # RISK DISTRIBUTION
    ##############################################################

    risk_df = pd.DataFrame({

        "Category":[

            "Current Risk",

            "Remaining"

        ],

        "Value":[

            climate_risk,

            100-climate_risk

        ]

    })

    fig_risk = px.pie(

        risk_df,

        names="Category",

        values="Value",

        hole=0.60,

        color_discrete_sequence=[

            "#d62728",

            "#d9d9d9"

        ]

    )

    fig_risk.update_layout(

        title="Climate Risk Distribution",

        height=420

    )

    st.plotly_chart(

        fig_risk,

        use_container_width=True

    )
    ##############################################################
    # MITIGATION RECOMMENDATION
    ##############################################################

    st.markdown("---")

    st.subheader("🛡 Mitigation Recommendation")

    recommendation = {

        "Flood":[

            "✔ Improve drainage infrastructure.",

            "✔ Monitor river water level continuously.",

            "✔ Prepare evacuation routes.",

            "✔ Activate flood early warning system.",

            "✔ Coordinate with local disaster management agency."

        ],

        "Drought":[

            "✔ Implement water conservation.",

            "✔ Optimize irrigation management.",

            "✔ Increase groundwater monitoring.",

            "✔ Reduce non-essential water consumption.",

            "✔ Prepare emergency water supply."

        ],

        "Heatwave":[

            "✔ Reduce outdoor activities.",

            "✔ Increase drinking water intake.",

            "✔ Prepare cooling shelters.",

            "✔ Protect elderly and children.",

            "✔ Monitor heat index continuously."

        ],

        "Storm":[

            "✔ Monitor BMKG weather updates.",

            "✔ Secure vulnerable infrastructure.",

            "✔ Prepare emergency response teams.",

            "✔ Suspend marine activities if necessary.",

            "✔ Inspect electrical installations."

        ],

        "Normal":[

            "✔ Continue routine monitoring.",

            "✔ Maintain climate observation.",

            "✔ Update disaster preparedness plan.",

            "✔ Maintain Early Warning System.",

            "✔ Continue environmental awareness."

        ]

    }

    if label in recommendation:

        for rec in recommendation[label]:

            st.write(rec)

    ##############################################################
    # EXECUTIVE CONCLUSION
    ##############################################################

    st.markdown("---")

    st.subheader("📑 Executive Conclusion")

    st.success(f"""

### Climate Intelligence Summary

Machine Learning analysis indicates that the predicted climate event is:

## **{label}**

Prediction Confidence:

**{risk:.2f}%**

Climate Risk Level:

**{status}**

The XGBoost model successfully integrates climate variables to support
Decision Support Systems (DSS) and Early Warning Systems (EWS)
for extreme climate events.

The most influential predictor according to the model is:

**{importance.iloc[0]['Feature']}**

The system recommends implementing mitigation strategies according
to the predicted disaster category.

""")
    ##############################################################
    # EXPORT REPORT
    ##############################################################

    st.markdown("---")

    st.subheader("📥 Export Executive Report")

    report = pd.DataFrame({

        "Parameter":[

            "Prediction",

            "Confidence (%)",

            "Risk Level",

            "Climate Risk Index",

            "Top Feature",

            "Model",

            "Observations",

            "Features"

        ],

        "Value":[

            label,

            round(risk,2),

            status,

            round(climate_risk,2),

            importance.iloc[0]["Feature"],

            "XGBoost",

            len(df),

            len(feature_columns)

        ]

    })

    ##############################################################
    # SHOW REPORT
    ##############################################################

    st.dataframe(

        report,

        use_container_width=True,

        hide_index=True

    )

    ##############################################################
    # DOWNLOAD CSV
    ##############################################################

    csv = report.to_csv(index=False)

    st.download_button(

        "📥 Download Executive Report (CSV)",

        csv,

        file_name="Executive_Report.csv",

        mime="text/csv"

    )

    ##############################################################
    # DOWNLOAD JSON
    ##############################################################

    json = report.to_json(

        orient="records",

        indent=4

    )

    st.download_button(

        "📥 Download Executive Report (JSON)",

        json,

        file_name="Executive_Report.json",

        mime="application/json"

    )

    ##############################################################
    # DASHBOARD FOOTER
    ##############################################################

    st.markdown("---")

    st.markdown("""

<div style="text-align:center;padding:20px">

<h3>🌍 Climate Intelligence Platform</h3>

<b>Machine Learning • Explainable AI • Decision Support System</b>

<br><br>

Developed by

<b>Melly Ariska</b>

<br>

Physics Education Department

Faculty of Teacher Training and Education

Universitas Sriwijaya

<br><br>

Version 1.0 | 2026

</div>

""",

unsafe_allow_html=True)