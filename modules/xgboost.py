##############################################################
# modules/xgboost.py
# XGBoost Machine Learning Dashboard
# Author : Melly Ariska
##############################################################

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from xgboost import XGBClassifier


##############################################################
# MAIN FUNCTION
##############################################################

def xgboost_dashboard(df):

    st.header("🚀 XGBoost Machine Learning")

    st.markdown("""
Dashboard ini digunakan untuk melakukan klasifikasi
kejadian iklim ekstrem menggunakan algoritma
Extreme Gradient Boosting (XGBoost).
""")
        ##############################################################
    # FEATURE
    ##############################################################

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

    ##############################################################
    # PREPROCESSING
    ##############################################################

    X = df[feature_columns].copy()

    for col in feature_columns:
        X[col] = pd.to_numeric(
            X[col],
            errors="coerce"
        )

    df["Disaster"] = (
        df["Disaster"]
        .fillna("Normal")
        .astype(str)
        .str.strip()
    )

    y = df["Disaster"]

    data = pd.concat([X, y], axis=1)

    data = data.dropna().reset_index(drop=True)

    X = data[feature_columns]

    y = data["Disaster"]

    encoder = LabelEncoder()

    y = encoder.fit_transform(y)

    st.success(
        f"Dataset siap digunakan ({len(data)} observasi)"
    )
        ##############################################################
    # HYPERPARAMETER
    ##############################################################

    st.markdown("---")
    st.subheader("⚙ Hyperparameter")

    col1, col2 = st.columns(2)

    with col1:

        n_estimators = st.slider(
            "Number of Trees",
            min_value=50,
            max_value=500,
            value=200,
            step=10
        )

        max_depth = st.slider(
            "Maximum Depth",
            min_value=2,
            max_value=15,
            value=6
        )

    with col2:

        learning_rate = st.slider(
            "Learning Rate",
            min_value=0.01,
            max_value=0.50,
            value=0.10,
            step=0.01
        )

        subsample = st.slider(
            "Subsample",
            min_value=0.50,
            max_value=1.00,
            value=0.80,
            step=0.05
        )

    ##############################################################
    # TRAIN TEST SPLIT
    ##############################################################

    st.markdown("---")

    if len(np.unique(y)) < 2:

        st.error(
            "Dataset hanya memiliki satu kelas sehingga model tidak dapat dilatih."
        )

        return

    try:

        X_train, X_test, y_train, y_test = train_test_split(

            X,
            y,

            test_size=0.30,

            random_state=42,

            stratify=y

        )

    except Exception as e:

        st.warning(
            f"Stratified split gagal ({e}), menggunakan random split."
        )

        X_train, X_test, y_train, y_test = train_test_split(

            X,
            y,

            test_size=0.30,

            random_state=42

        )

    ##############################################################
    # DATA INFORMATION
    ##############################################################

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Training Data",
        len(X_train)
    )

    c2.metric(
        "Testing Data",
        len(X_test)
    )

    c3.metric(
        "Number of Classes",
        len(np.unique(y))
    )
        ##############################################################
    # MODEL CONFIGURATION
    ##############################################################

    st.markdown("---")
    st.subheader("🚀 XGBoost Training")

    model = XGBClassifier(

        objective="multi:softprob",

        num_class=len(np.unique(y)),

        n_estimators=n_estimators,

        max_depth=max_depth,

        learning_rate=learning_rate,

        subsample=subsample,

        colsample_bytree=0.80,

        random_state=42,

        eval_metric="mlogloss",

        n_jobs=-1

    )

    ##############################################################
    # TRAIN BUTTON
    ##############################################################

    train = st.button(
        "🚀 Train XGBoost Model",
        use_container_width=True
    )

    if not train:

        st.info(
            "Klik tombol **Train XGBoost Model** untuk memulai proses training."
        )

        return

    ##############################################################
    # TRAINING
    ##############################################################

    with st.spinner("Training XGBoost Model..."):

        model.fit(

            X_train,

            y_train

        )

    st.success("✅ Training selesai.")

    ##############################################################
    # PREDICTION
    ##############################################################

    pred = model.predict(X_test)

    prob = model.predict_proba(X_test)

    ##############################################################
    # PERFORMANCE
    ##############################################################

    acc = accuracy_score(

        y_test,

        pred

    )

    precision = precision_score(

        y_test,

        pred,

        average="weighted",

        zero_division=0

    )

    recall = recall_score(

        y_test,

        pred,

        average="weighted",

        zero_division=0

    )

    f1 = f1_score(

        y_test,

        pred,

        average="weighted",

        zero_division=0

    )
        ##############################################################
    # KPI DASHBOARD
    ##############################################################

    st.markdown("---")
    st.subheader("📊 Model Performance")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Accuracy",
        f"{acc:.4f}"
    )

    c2.metric(
        "Precision",
        f"{precision:.4f}"
    )

    c3.metric(
        "Recall",
        f"{recall:.4f}"
    )

    c4.metric(
        "F1 Score",
        f"{f1:.4f}"
    )

    ##############################################################
    # CONFUSION MATRIX
    ##############################################################

    st.markdown("---")
    st.subheader("🔥 Confusion Matrix")

    cm = confusion_matrix(
        y_test,
        pred
    )

    cm_df = pd.DataFrame(
        cm,
        index=encoder.classes_,
        columns=encoder.classes_
    )

    fig_cm = px.imshow(

        cm_df,

        text_auto=True,

        color_continuous_scale="Blues",

        aspect="auto"

    )

    fig_cm.update_layout(

        title="Confusion Matrix",

        height=600

    )

    st.plotly_chart(

        fig_cm,

        use_container_width=True

    )

    ##############################################################
    # CLASSIFICATION REPORT
    ##############################################################

    st.markdown("---")
    st.subheader("📋 Classification Report")

    report = classification_report(

        y_test,

        pred,

        target_names=encoder.classes_,

        output_dict=True,

        zero_division=0

    )

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(

        report_df,

        use_container_width=True

    )
        ##############################################################
    # FEATURE IMPORTANCE
    ##############################################################

    st.markdown("---")
    st.subheader("🌳 Feature Importance")

    importance = pd.DataFrame(

        {

            "Feature": feature_columns,

            "Importance": model.feature_importances_

        }

    )

    importance = importance.sort_values(

        by="Importance",

        ascending=False

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

        color_continuous_scale="Viridis"

    )

    fig_bar.update_layout(

        title="XGBoost Feature Importance",

        height=600

    )

    st.plotly_chart(

        fig_bar,

        use_container_width=True

    )

    ##############################################################
    # TOP 10 FEATURES
    ##############################################################

    st.subheader("🏆 Top 10 Important Variables")

    top10 = importance.head(10)

    st.dataframe(

        top10,

        use_container_width=True

    )

    ##############################################################
    # PIE CHART
    ##############################################################

    fig_pie = px.pie(

        top10,

        names="Feature",

        values="Importance",

        hole=0.45,

        color_discrete_sequence=px.colors.qualitative.Set2

    )

    fig_pie.update_layout(

        title="Contribution of Top Variables",

        height=500

    )

    st.plotly_chart(

        fig_pie,

        use_container_width=True

    )

    ##############################################################
    # DOWNLOAD FEATURE IMPORTANCE
    ##############################################################

    csv = importance.to_csv(index=False)

    st.download_button(

        "📥 Download Feature Importance",

        data=csv,

        file_name="feature_importance_xgboost.csv",

        mime="text/csv"

    )
        ##############################################################
    # PREDICTION RESULT
    ##############################################################

    st.markdown("---")
    st.subheader("🔮 Prediction Result")

    result = pd.DataFrame({

        "Actual":
            encoder.inverse_transform(y_test),

        "Prediction":
            encoder.inverse_transform(pred)

    })

    st.dataframe(

        result,

        use_container_width=True

    )

    ##############################################################
    # PREDICTION DISTRIBUTION
    ##############################################################

    st.markdown("---")
    st.subheader("📊 Prediction Distribution")

    count = result["Prediction"].value_counts()

    fig = px.pie(

        names=count.index,

        values=count.values,

        hole=0.45,

        color_discrete_sequence=px.colors.qualitative.Set2

    )

    fig.update_layout(

        height=450

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    ##############################################################
    # ACCURACY GAUGE
    ##############################################################

    st.markdown("---")
    st.subheader("🎯 Model Accuracy Gauge")

    gauge = go.Figure(

        go.Indicator(

            mode="gauge+number",

            value=acc*100,

            title={"text":"Accuracy (%)"},

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

    gauge.update_layout(

        height=420

    )

    st.plotly_chart(

        gauge,

        use_container_width=True

    )

    ##############################################################
    # SAVE MODEL
    ##############################################################

    st.markdown("---")

    st.session_state["xgb_model"] = model
    st.session_state["encoder"] = encoder
    st.session_state["feature_columns"] = feature_columns
    st.session_state["X_train"] = X_train
    st.session_state["X_test"] = X_test
    st.session_state["y_train"] = y_train
    st.session_state["y_test"] = y_test

    os.makedirs("models", exist_ok=True)

    model_data = {
        "model": model,
        "encoder": encoder,
        "features": feature_columns,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test
    }

    joblib.dump(model_data, "models/xgboost.pkl")

    if os.path.exists("models/xgboost.pkl"):
        st.success("✅ XGBoost model saved successfully.")

        with open("models/xgboost.pkl","rb") as f:
            st.download_button(
                "💾 Download XGBoost Model",
                data=f,
                file_name="xgboost.pkl",
                mime="application/octet-stream"
            )

    csv = result.to_csv(index=False)

    st.download_button(
        "📥 Download Prediction Result",
        csv,
        file_name="prediction_result.csv",
        mime="text/csv"
    )

    st.markdown("---")

    st.success(f"""
### 🚀 XGBoost Training Completed Successfully

**Training Samples :** {len(X_train)}
**Testing Samples :** {len(X_test)}
**Number of Features :** {len(feature_columns)}
**Number of Classes :** {len(encoder.classes_)}

**Accuracy :** {acc:.4f}
**Precision :** {precision:.4f}
**Recall :** {recall:.4f}
**F1 Score :** {f1:.4f}

Model berhasil disimpan sebagai:
models/xgboost.pkl
""")
