import os
import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

import joblib


def random_forest_dashboard(df):

    st.header("🌳 Random Forest Machine Learning")

    ############################################################
    # FEATURES
    ############################################################

    feats = [
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
        "SPI",
    ]

    ############################################################
    # PREPROCESSING
    ############################################################

    # Pastikan fitur numerik
    X = df[feats].copy()

    for col in feats:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    # Bersihkan target
    df["Disaster"] = (
        df["Disaster"]
        .fillna("Normal")
        .astype(str)
        .str.strip()
    )

    y = df["Disaster"]

    # Gabungkan lalu hapus NaN
    data = pd.concat([X, y], axis=1)

    data = data.dropna().reset_index(drop=True)

    X = data[feats]
    y = data["Disaster"]
    st.write("Missing values")
    st.write(data.isna().sum())
    st.write("Distribusi kelas")
    st.write(y.value_counts(dropna=False))
    st.info(f"Jumlah data yang digunakan: {len(data)}")

    ############################################################
    # PARAMETER
    ############################################################

    c1, c2 = st.columns(2)

    with c1:
        n = st.slider(
            "Number of Trees",
            50,
            500,
            200,
            10,
        )

    with c2:
        d = st.slider(
            "Maximum Depth",
            2,
            30,
            10,
        )

    ############################################################
    # TRAIN TEST SPLIT
    ############################################################

    # Jika hanya satu kelas maka hentikan
    if len(y.unique()) < 2:
        st.error("Dataset hanya memiliki satu kelas.")
        return

    # Jika ada kelas hanya 1 sampel maka jangan stratify
    if y.value_counts().min() < 2:

        st.warning(
            "Ada kelas yang hanya memiliki satu sampel. Stratify dinonaktifkan."
        )

        Xtr, Xte, ytr, yte = train_test_split(
            X,
            y,
            test_size=0.30,
            random_state=42,
        )

    else:

        Xtr, Xte, ytr, yte = train_test_split(
            X,
            y,
            test_size=0.30,
            random_state=42,
            stratify=y,
        )

    ############################################################
    # MODEL
    ############################################################

    model = RandomForestClassifier(
        n_estimators=n,
        max_depth=d,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(Xtr, ytr)

    pred = model.predict(Xte)

    ############################################################
    # METRICS
    ############################################################

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Accuracy",
        f"{accuracy_score(yte, pred):.3f}",
    )

    c2.metric(
        "Precision",
        f"{precision_score(yte, pred, average='weighted', zero_division=0):.3f}",
    )

    c3.metric(
        "Recall",
        f"{recall_score(yte, pred, average='weighted', zero_division=0):.3f}",
    )

    c4.metric(
        "F1 Score",
        f"{f1_score(yte, pred, average='weighted', zero_division=0):.3f}",
    )

    ############################################################
    # FEATURE IMPORTANCE
    ############################################################

    st.subheader("Feature Importance")

    imp = pd.DataFrame(
        {
            "Feature": feats,
            "Importance": model.feature_importances_,
        }
    )

    imp = imp.sort_values(
        "Importance",
        ascending=False,
    )

    fig = px.bar(
        imp,
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale="Viridis",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    ############################################################
    # CONFUSION MATRIX
    ############################################################

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(
        yte,
        pred,
        labels=model.classes_,
    )

    fig2 = px.imshow(
        cm,
        x=model.classes_,
        y=model.classes_,
        text_auto=True,
        color_continuous_scale="Blues",
    )

    st.plotly_chart(
        fig2,
        use_container_width=True,
    )

    ############################################################
    # CLASSIFICATION REPORT
    ############################################################

    st.subheader("Classification Report")

    report = pd.DataFrame(
        classification_report(
            yte,
            pred,
            output_dict=True,
            zero_division=0,
        )
    ).transpose()

    st.dataframe(
        report,
        use_container_width=True,
    )

    ############################################################
    # SAVE MODEL
    ############################################################

    os.makedirs(
        "models",
        exist_ok=True,
    )

    model_path = "models/random_forest.pkl"

    joblib.dump(
        model,
        model_path,
    )

    with open(model_path, "rb") as f:

        st.download_button(
            "📥 Download Random Forest Model",
            f,
            file_name="random_forest.pkl",
        )

    ############################################################
    # DOWNLOAD FEATURE IMPORTANCE
    ############################################################

    st.download_button(
        "📥 Download Feature Importance",
        imp.to_csv(index=False),
        file_name="feature_importance.csv",
        mime="text/csv",
    )
