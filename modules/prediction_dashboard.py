##############################################################
# modules/prediction_dashboard.py
# Prediction Dashboard
##############################################################

import os
import joblib
import numpy as np
import pandas as pd

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


##############################################################
# MAIN FUNCTION
##############################################################

def prediction_dashboard():

    st.header("🔮 Climate Disaster Prediction")

    st.markdown("""
Prediksi jenis bencana iklim menggunakan model
**XGBoost Machine Learning** yang telah dilatih.

Masukkan parameter iklim kemudian klik tombol
**Predict Disaster**.
""")

    ##############################################################
    # LOAD MODEL
    ##############################################################

    model_path = "models/xgboost.pkl"

    if not os.path.exists(model_path):

        st.error("Model XGBoost belum tersedia.")

        st.info("Silakan buka menu **XGBoost** dan lakukan training model terlebih dahulu.")

        return

    model_file = joblib.load(model_path)

    model = model_file["model"]

    encoder = model_file["encoder"]

    feature_columns = model_file["features"]

    st.success("✅ Model berhasil dimuat.")

    st.write("Jumlah Feature :", len(feature_columns))
        ##############################################################
    # INPUT CLIMATE VARIABLES
    ##############################################################

    st.markdown("---")
    st.subheader("🌦 Climate Variables")

    col1, col2 = st.columns(2)

    with col1:

        rainfall = st.number_input(
            "Rainfall (mm)",
            value=120.0
        )

        tmax = st.number_input(
            "Maximum Temperature (°C)",
            value=31.0
        )

        tmin = st.number_input(
            "Minimum Temperature (°C)",
            value=24.0
        )

        humidity = st.number_input(
            "Humidity (%)",
            value=82.0
        )

        pressure = st.number_input(
            "Pressure (hPa)",
            value=1010.0
        )

        wind = st.number_input(
            "Wind Speed (m/s)",
            value=3.5
        )

    with col2:

        solar = st.number_input(
            "Solar Radiation",
            value=220.0
        )

        enso = st.number_input(
            "ENSO Index",
            value=0.20
        )

        iod = st.number_input(
            "IOD Index",
            value=0.10
        )

        ndvi = st.number_input(
            "NDVI",
            value=0.65
        )

        spi = st.number_input(
            "SPI",
            value=-0.50
        )

    ##############################################################
    # BUILD INPUT DATAFRAME
    ##############################################################

    input_data = pd.DataFrame({

        "Rainfall":[rainfall],
        "Tmax":[tmax],
        "Tmin":[tmin],
        "Humidity":[humidity],
        "Pressure":[pressure],
        "Wind":[wind],
        "Solar":[solar],
        "ENSO":[enso],
        "IOD":[iod],
        "NDVI":[ndvi],
        "SPI":[spi]

    })

    st.markdown("---")

    st.subheader("📋 Input Data")

    st.dataframe(

        input_data,

        use_container_width=True,

        hide_index=True

    )
        ##############################################################
    # PREDICT BUTTON
    ##############################################################

    st.markdown("---")

    predict = st.button(

        "🚀 Predict Disaster",

        use_container_width=True

    )

    if not predict:

        st.info(

            "Klik tombol **Predict Disaster** untuk menjalankan model."

        )

        return

    ##############################################################
    # PREDICTION
    ##############################################################

    with st.spinner("Running XGBoost Prediction..."):

        prediction = model.predict(input_data)

        probability = model.predict_proba(input_data)

    ##############################################################
    # DECODE LABEL
    ##############################################################

    predicted_class = encoder.inverse_transform(prediction)[0]

    probability = probability[0]

    st.success("✅ Prediction Completed Successfully")
        ##############################################################
    # PREDICTION RESULT
    ##############################################################

    st.markdown("---")

    st.subheader("🎯 Prediction Result")

    st.metric(

        "Predicted Disaster",

        predicted_class

    )

    ##############################################################
    # PROBABILITY TABLE
    ##############################################################

    probability_df = pd.DataFrame({

        "Disaster": encoder.classes_,

        "Probability": probability

    })

    probability_df["Probability (%)"] = (

        probability_df["Probability"] * 100

    ).round(2)

    probability_df = probability_df.sort_values(

        "Probability",

        ascending=False

    )

    st.subheader("📊 Prediction Probability")

    st.dataframe(

        probability_df,

        use_container_width=True,

        hide_index=True

    )

    ##############################################################
    # BAR CHART
    ##############################################################

    fig_bar = px.bar(

        probability_df,

        x="Probability (%)",

        y="Disaster",

        orientation="h",

        color="Probability (%)",

        color_continuous_scale="Turbo",

        text="Probability (%)"

    )

    fig_bar.update_layout(

        title="Prediction Probability",

        height=500

    )

    st.plotly_chart(

        fig_bar,

        use_container_width=True

    )

    ##############################################################
    # PIE CHART
    ##############################################################

    fig_pie = px.pie(

        probability_df,

        names="Disaster",

        values="Probability (%)",

        hole=0.45,

        color_discrete_sequence=px.colors.qualitative.Set2

    )

    fig_pie.update_layout(

        title="Probability Distribution",

        height=500

    )

    st.plotly_chart(

        fig_pie,

        use_container_width=True

    )
        ##############################################################
    # RISK LEVEL
    ##############################################################

    st.markdown("---")
    st.subheader("🚨 Disaster Risk Level")

    risk = float(probability.max() * 100)

    if risk < 40:

        level = "LOW"
        color = "green"

    elif risk < 70:

        level = "MODERATE"
        color = "orange"

    else:

        level = "HIGH"
        color = "red"

    ##############################################################
    # KPI
    ##############################################################

    c1, c2 = st.columns(2)

    c1.metric(

        "Risk Level",

        level

    )

    c2.metric(

        "Highest Probability",

        f"{risk:.2f}%"

    )

    ##############################################################
    # GAUGE
    ##############################################################

    gauge = go.Figure(

        go.Indicator(

            mode="gauge+number",

            value=risk,

            title={"text":"Disaster Risk (%)"},

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

        height=450

    )

    st.plotly_chart(

        gauge,

        use_container_width=True

    )
        ##############################################################
    # RECOMMENDATION SYSTEM
    ##############################################################

    st.markdown("---")
    st.subheader("💡 Decision Support Recommendation")

    recommendation = {

        "Flood":[

            "✔ Improve drainage systems.",

            "✔ Monitor river water levels.",

            "✔ Prepare flood evacuation routes.",

            "✔ Activate Early Warning System."

        ],

        "Drought":[

            "✔ Implement water conservation.",

            "✔ Optimize irrigation scheduling.",

            "✔ Increase groundwater monitoring.",

            "✔ Encourage efficient water usage."

        ],

        "Heatwave":[

            "✔ Reduce outdoor activities.",

            "✔ Increase drinking water consumption.",

            "✔ Prepare cooling shelters.",

            "✔ Protect vulnerable populations."

        ],

        "Storm":[

            "✔ Monitor BMKG weather forecasts.",

            "✔ Secure infrastructure.",

            "✔ Prepare emergency response teams.",

            "✔ Minimize marine activities."

        ],

        "Normal":[

            "✔ Climate conditions are stable.",

            "✔ Continue routine monitoring.",

            "✔ Maintain preparedness programs."

        ]

    }

    if predicted_class in recommendation:

        st.success(

            f"Recommended Actions for **{predicted_class}**"

        )

        for item in recommendation[predicted_class]:

            st.write(item)

    else:

        st.info(

            "No recommendation available for this prediction."

        )

    ##############################################################
    # PREDICTION SUMMARY
    ##############################################################

    st.markdown("---")

    summary = pd.DataFrame({

        "Parameter":[

            "Predicted Disaster",

            "Risk Level",

            "Highest Probability (%)"

        ],

        "Value":[

            predicted_class,

            level,

            round(risk,2)

        ]

    })

    st.subheader("📋 Prediction Summary")

    st.dataframe(

        summary,

        use_container_width=True,

        hide_index=True

    )
        ##############################################################
    # DOWNLOAD RESULT
    ##############################################################

    st.markdown("---")
    st.subheader("📥 Export Prediction Result")

    ##############################################################
    # RESULT DATAFRAME
    ##############################################################

    result_df = pd.DataFrame({

        "Predicted_Disaster":[predicted_class],
        "Risk_Level":[level],
        "Highest_Probability(%)":[round(risk,2)]

    })

    ##############################################################
    # ADD PROBABILITY
    ##############################################################

    for i, cls in enumerate(encoder.classes_):

        result_df[f"Probability_{cls}"] = [

            round(probability[i]*100,2)

        ]

    ##############################################################
    # SHOW TABLE
    ##############################################################

    st.dataframe(

        result_df,

        use_container_width=True,

        hide_index=True

    )

    ##############################################################
    # DOWNLOAD CSV
    ##############################################################

    csv = result_df.to_csv(index=False)

    st.download_button(

        "📥 Download Prediction CSV",

        data=csv,

        file_name="prediction_result.csv",

        mime="text/csv"

    )

    ##############################################################
    # DOWNLOAD JSON
    ##############################################################

    json = result_df.to_json(

        orient="records",

        indent=4

    )

    st.download_button(

        "📥 Download Prediction JSON",

        data=json,

        file_name="prediction_result.json",

        mime="application/json"

    )
        ##############################################################
    # DASHBOARD SUMMARY
    ##############################################################

    st.markdown("---")

    st.subheader("📊 Prediction Dashboard Summary")

    c1, c2, c3 = st.columns(3)

    c1.metric(

        "Predicted Disaster",

        predicted_class

    )

    c2.metric(

        "Risk Level",

        level

    )

    c3.metric(

        "Prediction Confidence",

        f"{risk:.2f}%"

    )

    ##############################################################
    # INTERPRETATION
    ##############################################################

    st.markdown("---")

    st.success(f"""

## Interpretation

Model XGBoost berhasil melakukan prediksi kondisi iklim.

### Prediction Result

**Predicted Disaster :**
{predicted_class}

**Confidence :**
{risk:.2f}%

**Risk Level :**
{level}

### Decision Support

Dashboard ini dapat digunakan sebagai sistem pendukung
keputusan (Decision Support System) dalam mitigasi
bencana hidrometeorologi berbasis Machine Learning.

""")

    ##############################################################
    # FOOTER
    ##############################################################

    st.markdown("---")

    st.caption("""

🌦 Climate Disaster Prediction Dashboard

Machine Learning Model :
**Extreme Gradient Boosting (XGBoost)**

Prediction Module :
**Decision Support System**

Developed by

**Melly Ariska**

Physics Education Department

Faculty of Teacher Training and Education

Universitas Sriwijaya

2026

""")
