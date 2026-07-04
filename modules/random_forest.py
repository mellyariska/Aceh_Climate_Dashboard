##############################################################
# modules/random_forest.py
##############################################################

import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import joblib

##############################################################
# RANDOM FOREST DASHBOARD
##############################################################

def random_forest_dashboard(df):

    st.header("🌳 Random Forest Machine Learning")

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

    target = "Disaster"

    ###########################################################
    # DATA
    ###########################################################

    X = df[feature_columns]

    y = df[target]

    ###########################################################
    # PARAMETER
    ###########################################################

    col1,col2 = st.columns(2)

    with col1:

        n_tree = st.slider(

            "Number of Trees",

            50,

            500,

            200,

            10

        )

    with col2:

        depth = st.slider(

            "Maximum Depth",

            2,

            30,

            10

        )

    ###########################################################
    # TRAIN TEST
    ###########################################################

    X_train,X_test,y_train,y_test = train_test_split(

        X,

        y,

        test_size=0.30,

        random_state=42,

        stratify=y

    )

    ###########################################################
    # MODEL
    ###########################################################

    model = RandomForestClassifier(

        n_estimators=n_tree,

        max_depth=depth,

        random_state=42

    )

    model.fit(

        X_train,

        y_train

    )

    ###########################################################
    # PREDICTION
    ###########################################################

    pred = model.predict(

        X_test

    )

    ###########################################################
    # METRIC
    ###########################################################

    acc = accuracy_score(

        y_test,

        pred

    )

    precision = precision_score(

        y_test,

        pred,

        average="weighted"

    )

    recall = recall_score(

        y_test,

        pred,

        average="weighted"

    )

    f1 = f1_score(

        y_test,

        pred,

        average="weighted"

    )

    ###########################################################
    # KPI
    ###########################################################

    c1,c2,c3,c4 = st.columns(4)

    c1.metric(

        "Accuracy",

        f"{acc:.3f}"

    )

    c2.metric(

        "Precision",

        f"{precision:.3f}"

    )

    c3.metric(

        "Recall",

        f"{recall:.3f}"

    )

    c4.metric(

        "F1 Score",

        f"{f1:.3f}"

    )

    ###########################################################
    # FEATURE IMPORTANCE
    ###########################################################

    st.subheader("Feature Importance")

    importance = pd.DataFrame(

        {

            "Feature":feature_columns,

            "Importance":model.feature_importances_

        }

    )

    importance = importance.sort_values(

        "Importance",

        ascending=False

    )

    fig = px.bar(

        importance,

        x="Importance",

        y="Feature",

        orientation="h",

        color="Importance",

        color_continuous_scale="Viridis"

    )

    fig.update_layout(

        height=500

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    ###########################################################
    # CONFUSION MATRIX
    ###########################################################

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(

        y_test,

        pred,

        labels=model.classes_

    )

    cm_df = pd.DataFrame(

        cm,

        index=model.classes_,

        columns=model.classes_

    )

    heat = px.imshow(

        cm_df,

        text_auto=True,

        color_continuous_scale="Blues"

    )

    heat.update_layout(

        height=600

    )

    st.plotly_chart(

        heat,

        use_container_width=True

    )

    ###########################################################
    # REPORT
    ###########################################################

    st.subheader("Classification Report")

    report = classification_report(

        y_test,

        pred,

        output_dict=True

    )

    report_df = pd.DataFrame(

        report

    ).transpose()

    st.dataframe(

        report_df,

        use_container_width=True

    )

    ###########################################################
    # DOWNLOAD MODEL
    ###########################################################

    joblib.dump(

        model,

        "models/random_forest.pkl"

    )

    with open(

        "models/random_forest.pkl",

        "rb"

    ) as f:

        st.download_button(

            "📥 Download Random Forest Model",

            f,

            file_name="random_forest.pkl"

        )

    ###########################################################
    # DOWNLOAD FEATURE IMPORTANCE
    ###########################################################

    csv = importance.to_csv(index=False)

    st.download_button(

        "📥 Download Feature Importance",

        csv,

        file_name="feature_importance.csv",

        mime="text/csv"

    )

    ###########################################################
    # INTERPRETATION
    ###########################################################

    st.markdown("---")

    st.success(f"""

Random Forest berhasil dilatih menggunakan:

• {len(feature_columns)} variabel iklim

• {len(df)} observasi

• Jumlah pohon = {n_tree}

• Kedalaman maksimum = {depth}

Model ini dapat digunakan sebagai baseline
untuk dibandingkan dengan XGBoost,
CatBoost, dan LSTM.

""")