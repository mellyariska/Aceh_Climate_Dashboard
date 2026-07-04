##############################################################
# modules/xgboost.py
# XGBoost Classification Dashboard
##############################################################

import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from xgboost import XGBClassifier

import joblib

##############################################################
# MAIN FUNCTION
##############################################################

def xgboost_dashboard(df):

    st.header("🚀 Extreme Gradient Boosting (XGBoost)")

    st.markdown("""
XGBoost digunakan untuk memprediksi kejadian iklim ekstrem
menggunakan data Climate Big Data yang telah diintegrasikan
melalui Climate Big Data Integration Framework (CBDIF).
""")

    ##########################################################
    # Feature Selection
    ##########################################################

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

    ##########################################################
    # Dataset
    ##########################################################

    X = df[feature_columns]

    y = df[target]

    ##########################################################
    # Encode Target
    ##########################################################

    encoder = LabelEncoder()

    y = encoder.fit_transform(y)

    ##########################################################
    # Sidebar Hyperparameter
    ##########################################################

    st.subheader("⚙ Hyperparameter")

    c1, c2 = st.columns(2)

    with c1:

        n_estimators = st.slider(

            "Number of Trees",

            50,

            500,

            200,

            step=10

        )

        max_depth = st.slider(

            "Maximum Depth",

            2,

            15,

            6

        )

    with c2:

        learning_rate = st.slider(

            "Learning Rate",

            0.01,

            0.50,

            0.10,

            step=0.01

        )

        subsample = st.slider(

            "Subsample",

            0.50,

            1.00,

            0.80,

            step=0.05

        )

    ##########################################################
    # Train Test Split
    ##########################################################

    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=0.30,

        random_state=42,

        stratify=y

    )

    ##########################################################
    # Information
    ##########################################################

    st.info(f"""

Training Data : {len(X_train)}

Testing Data : {len(X_test)}

Features : {len(feature_columns)}

Classes : {len(np.unique(y))}

""")

    ##########################################################
    # Model Configuration
    ##########################################################

    model = XGBClassifier(

        objective="multi:softmax",

        num_class=len(np.unique(y)),

        n_estimators=n_estimators,

        learning_rate=learning_rate,

        max_depth=max_depth,

        subsample=subsample,

        random_state=42,

        eval_metric="mlogloss"

    )

    ##########################################################
    # Tombol Training
    ##########################################################

    train = st.button("🚀 Train XGBoost Model")

    if train:

        with st.spinner("Training XGBoost Model ..."):

            model.fit(

                X_train,

                y_train

            )
####################################################
# CONFUSION MATRIX
####################################################

st.markdown("---")

st.subheader("Confusion Matrix")

cm = confusion_matrix(

    y_test,

    pred

)

cm_df = pd.DataFrame(

    cm,

    index=encoder.classes_,

    columns=encoder.classes_

)

fig = px.imshow(

    cm_df,

    text_auto=True,

    color_continuous_scale="Blues",

    aspect="auto"

)

fig.update_layout(

    height=600,

    title="Confusion Matrix"

)

st.plotly_chart(

    fig,

    use_container_width=True

)

####################################################
# CLASSIFICATION REPORT
####################################################

st.markdown("---")

st.subheader("Classification Report")

report = classification_report(

    y_test,

    pred,

    target_names=encoder.classes_,

    output_dict=True

)

report_df = pd.DataFrame(

    report

).transpose()

st.dataframe(

    report_df,

    use_container_width=True

)

####################################################
# FEATURE IMPORTANCE
####################################################

st.markdown("---")

st.subheader("Feature Importance")

importance = pd.DataFrame(

    {

        "Feature":feature_columns,

        "Importance":model.feature_importances_

    }

)

importance = importance.sort_values(

    by="Importance",

    ascending=False

)

bar = px.bar(

    importance,

    x="Importance",

    y="Feature",

    orientation="h",

    color="Importance",

    color_continuous_scale="Viridis"

)

bar.update_layout(

    height=600,

    title="XGBoost Feature Importance"

)

st.plotly_chart(

    bar,

    use_container_width=True

)

####################################################
# TOP 10 FEATURE
####################################################

st.subheader("Top Important Variables")

top10 = importance.head(10)

st.dataframe(

    top10,

    use_container_width=True

)

####################################################
# FEATURE IMPORTANCE PIE
####################################################

pie = px.pie(

    top10,

    names="Feature",

    values="Importance",

    hole=.45

)

pie.update_layout(

    height=500,

    title="Contribution of Top Variables"

)

st.plotly_chart(

    pie,

    use_container_width=True

)

####################################################
# DOWNLOAD FEATURE IMPORTANCE
####################################################

csv = importance.to_csv(

    index=False

)

st.download_button(

    "📥 Download Feature Importance",

    csv,

    file_name="feature_importance_xgboost.csv",

    mime="text/csv"

)

####################################################
# SAVE MODEL
####################################################

joblib.dump(

    {

        "model":model,

        "encoder":encoder,

        "features":feature_columns

    },

    "models/xgboost.pkl"

)

with open(

    "models/xgboost.pkl",

    "rb"

) as f:

    st.download_button(

        "💾 Download XGBoost Model",

        data=f,

        file_name="xgboost.pkl"

    )

####################################################
# SUMMARY
####################################################

st.markdown("---")

st.success(f"""

### XGBoost Training Completed

Training Samples : {len(X_train)}

Testing Samples : {len(X_test)}

Number of Features : {len(feature_columns)}

Model Accuracy : {acc:.4f}

The trained model has been saved successfully
and is ready for Explainable AI (SHAP),
Prediction Dashboard,
and Climate Decision Support System.

""")

####################################################
# Prediction
####################################################

pred = model.predict(X_test)

prob = model.predict_proba(X_test)

####################################################
# Evaluation
####################################################

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

####################################################
# KPI Dashboard
####################################################

st.markdown("---")

st.subheader("📊 Model Performance")

c1,c2,c3,c4 = st.columns(4)

with c1:

    st.metric(

        "Accuracy",

        f"{acc:.4f}"

    )

with c2:

    st.metric(

        "Precision",

        f"{precision:.4f}"

    )

with c3:

    st.metric(

        "Recall",

        f"{recall:.4f}"

    )

with c4:

    st.metric(

        "F1 Score",

        f"{f1:.4f}"

    )

####################################################
# Prediction Table
####################################################

st.markdown("---")

st.subheader("Prediction Result")

result = pd.DataFrame(

    {

        "Actual":

            encoder.inverse_transform(

                y_test

            ),

        "Prediction":

            encoder.inverse_transform(

                pred

            )

    }

)

st.dataframe(

    result,

    use_container_width=True

)

####################################################
# Prediction Distribution
####################################################

st.subheader("Prediction Distribution")

count = result["Prediction"].value_counts()

fig = px.pie(

    names=count.index,

    values=count.values,

    hole=.45,

    color_discrete_sequence=px.colors.qualitative.Set2

)

fig.update_layout(

    height=450

)

st.plotly_chart(

    fig,

    use_container_width=True

)

####################################################
# Accuracy Gauge
####################################################

gauge = go.Figure(

go.Indicator(

mode="gauge+number",

value=acc*100,

title={

"text":"Accuracy"

},

gauge={

"axis":{

"range":[0,100]

},

"bar":{

"color":"royalblue"

},

"steps":[

{

"range":[0,60],

"color":"red"

},

{

"range":[60,80],

"color":"orange"

},

{

"range":[80,90],

"color":"yellow"

},

{

"range":[90,100],

"color":"green"

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

####################################################
# Download Prediction
####################################################

csv = result.to_csv(

index=False

)

st.download_button(

"📥 Download Prediction",

csv,

file_name="prediction.csv",

mime="text/csv"

)
            st.success("Training Finished")

            ####################################################
            # Save Model
            ####################################################

            joblib.dump(

                {

                    "model": model,

                    "encoder": encoder

                },

                "models/xgboost.pkl"

            )
