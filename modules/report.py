##############################################################
# modules/report.py
# Climate Intelligence Executive Report
# Author : Melly Ariska
##############################################################

##############################################################
# IMPORT LIBRARY
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
# REPORT DASHBOARD
##############################################################

def report_dashboard():

    ##############################################################
    # PAGE HEADER
    ##############################################################

    st.header("📑 Climate Intelligence Executive Report")

    st.write("""
    Executive Report ini menyajikan ringkasan hasil analisis
    Machine Learning, Explainable Artificial Intelligence (SHAP),
    Decision Support System (DSS), serta Early Warning System (EWS)
    untuk prediksi bencana iklim di Provinsi Aceh.
    """)

    ##############################################################
    # LOAD XGBOOST MODEL
    ##############################################################

    model_path = "models/xgboost.pkl"

    if not os.path.exists(model_path):

        st.error("Model XGBoost belum tersedia.")

        st.info(
            "Silakan buka menu **XGBoost** dan lakukan training model terlebih dahulu."
        )

        return

    ##############################################################
    # LOAD MODEL
    ##############################################################

    model_file = joblib.load(model_path)

    model = model_file["model"]

    encoder = model_file["encoder"]

    feature_columns = model_file["features"]

    ##############################################################
    # LOAD DATASET
    ##############################################################

    dataset_path = "data/data_aceh_1985_2025.xlsx"

    if os.path.exists(dataset_path):

        df = pd.read_excel(dataset_path)

    elif os.path.exists("data_aceh_1985_2025.xlsx"):

        df = pd.read_excel("data_aceh_1985_2025.xlsx")

    else:

        st.error("Dataset tidak ditemukan.")

        return

    ##############################################################
    # CLEAN DATA
    ##############################################################

    df.columns = df.columns.str.strip()

    ##############################################################
    # CONVERT FEATURE TO NUMERIC
    ##############################################################

    for col in feature_columns:

        if col in df.columns:

            df[col] = pd.to_numeric(

                df[col],

                errors="coerce"

            )

    ##############################################################
    # HANDLE MISSING VALUE
    ##############################################################

    df[feature_columns] = df[feature_columns].fillna(

        df[feature_columns].mean()

    )

    ##############################################################
    # CLEAN TARGET
    ##############################################################

    if "Disaster" in df.columns:

        df["Disaster"] = (

            df["Disaster"]

            .fillna("Normal")

            .astype(str)

            .str.strip()

        )

    ##############################################################
    # SUCCESS
    ##############################################################

    st.success(

        f"Dataset berhasil dimuat ({len(df)} observasi)."

    )

    ##############################################################
    # EXECUTIVE SUMMARY
    ##############################################################

    st.markdown("---")

    st.subheader("📋 Executive Summary")

    col1,col2,col3 = st.columns([1,5,1])

    with col1:

        if os.path.exists("assets/logo_unsri.png"):

            st.image(

                "assets/logo_unsri.png",

                width=90

            )

    with col2:

        st.markdown("""

# 🌦 Climate Intelligence Platform

### Machine Learning • Explainable AI • Decision Support System

**Aceh Extreme Climate Dashboard**

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

    st.markdown("---")

    today = datetime.now()

    info1,info2 = st.columns(2)

    with info1:

        st.info(f"""

### Report Information

**Date**

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

    c1,c2,c3,c4 = st.columns(4)

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

        "Missing Value",

        int(df[feature_columns].isna().sum().sum())

    )

    ##############################################################
    # DATASET OVERVIEW
    ##############################################################

    st.markdown("---")

    st.subheader("🌍 Dataset Overview")

    overview = pd.DataFrame({

        "Information":[

            "Total Observation",

            "Total Variables",

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
    # DATA PREVIEW
    ##############################################################

    st.markdown("---")

    st.subheader("📊 Climate Dataset Preview")

    st.dataframe(

        df.head(15),

        use_container_width=True,

        hide_index=True

    )

    ##############################################################
    # DESCRIPTIVE STATISTICS
    ##############################################################

    st.markdown("---")

    st.subheader("📈 Descriptive Statistics")

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
        confusion_matrix,
        classification_report
    )

    ##############################################################
    # PREPARE DATA
    ##############################################################

    X = df[feature_columns].copy()

    ##############################################################
    # PREDICT
    ##############################################################

    try:

        pred = model.predict(X)

        prob = model.predict_proba(X)

    except Exception as e:

        st.error(f"Gagal melakukan prediksi : {e}")

        return

    ##############################################################
    # EVALUATION
    ##############################################################

    valid = df["Disaster"].isin(encoder.classes_)

   ##############################################################
# CLEAN TARGET LABEL
##############################################################

y = (
    df["Disaster"]
      .fillna("Normal")
      .astype(str)
      .str.strip()
)

##############################################################
# DEBUG LABEL
##############################################################

st.write("Label pada model :", list(encoder.classes_))
st.write("Label pada dataset :", sorted(y.unique()))

##############################################################
# FILTER LABEL VALID
##############################################################

valid = y.isin(encoder.classes_)

if valid.sum() == 0:

    st.error("Tidak ada label dataset yang cocok dengan model.")

    return

##############################################################
# GUNAKAN DATA YANG SUDAH BERSIH
##############################################################

X = X.loc[valid].reset_index(drop=True)
y = y.loc[valid].reset_index(drop=True)

##############################################################
# CEK APAKAH MASIH ADA DATA
##############################################################

if len(X) == 0:

    st.error("Tidak ada data valid untuk evaluasi model.")

    return

##############################################################
# ENCODE TARGET
##############################################################

try:

    y_true = encoder.transform(y)

except Exception:

    st.warning(
        "Label dataset tidak sepenuhnya sama dengan label model. Evaluasi dilewati."
    )

    acc = np.nan
    precision = np.nan
    recall = np.nan
    f1 = np.nan

else:

    ##############################################################
    # PREDIKSI ULANG
    ##############################################################

    y_pred = model.predict(X)

    ##############################################################
    # HITUNG METRIK
    ##############################################################

    acc = accuracy_score(
        y_true,
        y_pred
    )

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
    # PERFORMANCE BAR
    ##############################################################

    perf = pd.DataFrame({

        "Metric":[

            "Accuracy",

            "Precision",

            "Recall",

            "F1 Score"

        ],

        "Score":[

            acc,

            precision,

            recall,

            f1

        ]

    })

    fig = px.bar(

        perf,

        x="Metric",

        y="Score",

        color="Score",

        text_auto=".3f",

        color_continuous_scale="Viridis"

    )

    fig.update_layout(

        height=450

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    ##############################################################
    # GAUGE
    ##############################################################

    gauge = go.Figure(

        go.Indicator(

            mode="gauge+number",

            value=acc*100,

            title={"text":"Model Accuracy (%)"},

            gauge={

                "axis":{"range":[0,100]},

                "bar":{"color":"royalblue"},

                "steps":[

                    {"range":[0,60],"color":"#ffcccc"},

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
    # PREDICTION SUMMARY
    ##############################################################

    st.markdown("---")

    st.subheader("🌦 Prediction Summary")

    ##############################################################
    # LAST OBSERVATION
    ##############################################################

    latest = X.tail(1)

    prediction = model.predict(

        latest

    )[0]

    probability = model.predict_proba(

        latest

    )[0]

    predicted_class = encoder.inverse_transform(

        [prediction]

    )[0]

    ##############################################################
    # RISK
    ##############################################################

    confidence = float(

        probability.max()*100

    )

    if confidence < 40:

        risk_level = "LOW"

    elif confidence < 70:

        risk_level = "MODERATE"

    else:

        risk_level = "HIGH"

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
    # BAR
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

        height=500,

        title="Prediction Probability"

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    ##############################################################
    # PIE
    ##############################################################

    fig2 = px.pie(

        prob_df,

        names="Class",

        values="Probability (%)",

        hole=0.45,

        color_discrete_sequence=px.colors.qualitative.Set2

    )

    fig2.update_layout(

        height=450

    )

    st.plotly_chart(

        fig2,

        use_container_width=True

    )
        ##############################################################
    # FEATURE IMPORTANCE
    ##############################################################

    st.markdown("---")
    st.subheader("🌳 Feature Importance")

    ##############################################################
    # FEATURE IMPORTANCE
    ##############################################################

    importance = pd.DataFrame({

        "Feature": feature_columns,

        "Importance": model.feature_importances_

    })

    importance = importance.sort_values(

        by="Importance",

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

        "Climate Variables",

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

        title="XGBoost Feature Importance",

        height=600

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    ##############################################################
    # PIE CHART
    ##############################################################

    fig2 = px.pie(

        importance.head(10),

        names="Feature",

        values="Importance",

        hole=0.45,

        color_discrete_sequence=px.colors.qualitative.Set3

    )

    fig2.update_layout(

        title="Top 10 Important Variables",

        height=500

    )

    st.plotly_chart(

        fig2,

        use_container_width=True

    )

    ##############################################################
    # INTERPRETATION
    ##############################################################

    st.success(f"""

Top predictor berdasarkan model adalah

**{importance.iloc[0]['Feature']}**

dengan nilai importance sebesar

**{importance.iloc[0]['Importance']:.4f}**

Variabel tersebut memberikan kontribusi terbesar
terhadap prediksi bencana iklim.

""")
        ##############################################################
    # EARLY WARNING SYSTEM
    ##############################################################

    st.markdown("---")
    st.subheader("🚨 Early Warning System")

    ##############################################################
    # RISK LEVEL
    ##############################################################

    climate_risk = confidence

    if climate_risk < 40:

        status = "LOW"

        message = "🟢 Normal Climate Condition"

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

        "Risk Level",

        status

    )

    c3.metric(

        "Early Warning",

        message

    )

    ##############################################################
    # GAUGE
    ##############################################################

    gauge = go.Figure(

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

Climate conditions are relatively stable.

Routine monitoring is recommended.

""")

    elif status == "MODERATE":

        st.warning("""

### 🟡 MODERATE RISK

Potential extreme climate event detected.

Increase monitoring and preparedness.

""")

    else:

        st.error("""

### 🔴 HIGH RISK

High probability of extreme climate event.

Activate the Early Warning System immediately.

""")

    ##############################################################
    # RISK DISTRIBUTION
    ##############################################################

    risk_df = pd.DataFrame({

        "Category":[

            "Risk",

            "Safe"

        ],

        "Value":[

            climate_risk,

            100-climate_risk

        ]

    })

    fig = px.pie(

        risk_df,

        names="Category",

        values="Value",

        hole=0.60,

        color_discrete_sequence=[

            "#d62728",

            "#d9d9d9"

        ]

    )

    fig.update_layout(

        title="Climate Risk Distribution",

        height=420

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )
        ##############################################################
    # DECISION SUPPORT RECOMMENDATION
    ##############################################################

    st.markdown("---")
    st.subheader("🧭 Decision Support Recommendation")

    if status == "LOW":

        recommendation = [
            "Melakukan monitoring iklim secara rutin.",
            "Memperbarui database iklim bulanan.",
            "Melanjutkan kegiatan normal."
        ]

    elif status == "MODERATE":

        recommendation = [
            "Meningkatkan monitoring cuaca harian.",
            "Menyiapkan sumber daya mitigasi bencana.",
            "Memberikan informasi dini kepada masyarakat."
        ]

    else:

        recommendation = [
            "Mengaktifkan Early Warning System.",
            "Melakukan koordinasi dengan BPBD dan BMKG.",
            "Menyebarkan peringatan kepada masyarakat.",
            "Menyiapkan rencana evakuasi apabila diperlukan."
        ]

    for i, rec in enumerate(recommendation, start=1):

        st.write(f"**{i}.** {rec}")

    ##############################################################
    # DECISION SUPPORT SCORE
    ##############################################################

    st.markdown("---")
    st.subheader("📊 Decision Support Score")

    score = round(climate_risk,2)

    fig = go.Figure(

        go.Indicator(

            mode="number+delta",

            value=score,

            number={"suffix":" %"},

            title={"text":"Climate Decision Score"},

            delta={

                "reference":50,

                "relative":False

            }

        )

    )

    fig.update_layout(height=300)

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    ##############################################################
    # SUMMARY BOX
    ##############################################################

    st.success(f"""

### Executive Conclusion

**Prediction**

: **{predicted_class}**

**Confidence**

: **{confidence:.2f}%**

**Climate Risk**

: **{status}**

Dashboard merekomendasikan agar pengambil keputusan
menggunakan hasil prediksi Machine Learning sebagai
alat bantu dalam penyusunan strategi mitigasi bencana
dan perencanaan adaptasi perubahan iklim.

""")
        ##############################################################
    # DOWNLOAD REPORT
    ##############################################################

    st.markdown("---")
    st.subheader("💾 Export Report")

    ##############################################################
    # SUMMARY TABLE
    ##############################################################

    summary = pd.DataFrame({

        "Parameter":[

            "Prediction",

            "Confidence",

            "Risk Level",

            "Accuracy",

            "Precision",

            "Recall",

            "F1 Score"

        ],

        "Value":[

            predicted_class,

            round(confidence,2),

            status,

            round(acc,4),

            round(precision,4),

            round(recall,4),

            round(f1,4)

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

        file_name="executive_report.csv",

        mime="text/csv"

    )

    ##############################################################
    # DOWNLOAD FEATURE IMPORTANCE
    ##############################################################

    csv2 = importance.to_csv(index=False)

    st.download_button(

        "📥 Download Feature Importance",

        data=csv2,

        file_name="feature_importance.csv",

        mime="text/csv"

    )

    ##############################################################
    # FOOTER
    ##############################################################

    st.markdown("---")

    st.caption("""

Aceh Extreme Climate Dashboard

Machine Learning • XGBoost • Explainable AI • Decision Support System

Developed by **Melly Ariska**

Universitas Sriwijaya

© 2026

""")
