##############################################################
# modules/prediction_dashboard.py
##############################################################

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
##############################################################

##############################################################
# EARLY WARNING ENGINE
##############################################################

def calculate_climate_risk(

    rainfall,
    temperature,
    humidity,
    spi,
    enso,
    iod

):

    score = 0

    ##########################################################
    # Rainfall
    ##########################################################

    if rainfall >= 250:

        score += 30

    elif rainfall >= 180:

        score += 20

    elif rainfall >= 120:

        score += 10

    ##########################################################
    # Temperature
    ##########################################################

    if temperature >= 35:

        score += 20

    elif temperature >= 33:

        score += 10

    ##########################################################
    # Humidity
    ##########################################################

    if humidity >= 90:

        score += 15

    elif humidity <= 60:

        score += 10

    ##########################################################
    # SPI
    ##########################################################

    if spi >= 2:

        score += 20

    elif spi <= -2:

        score += 20

    ##########################################################
    # ENSO
    ##########################################################

    if abs(enso) >= 1.5:

        score += 10

    ##########################################################
    # IOD
    ##########################################################

    if abs(iod) >= 1.0:

        score += 5

    return min(score,100)

# MAIN
##############################################################

def prediction_dashboard():

    st.header("🌦 Extreme Climate Prediction")

    st.markdown("""
Masukkan parameter iklim kemudian tekan tombol
**Predict** untuk mengetahui prediksi kondisi
iklim ekstrem menggunakan model XGBoost.
""")

    ##########################################################
    # LOAD MODEL
    ##########################################################

    model_file = joblib.load(

        "models/xgboost.pkl"

    )

    model = model_file["model"]

    encoder = model_file["encoder"]

    ##########################################################
    # INPUT
    ##########################################################

    c1,c2,c3 = st.columns(3)

    with c1:

        rainfall = st.number_input(

            "Rainfall (mm)",

            0.0,

            600.0,

            180.0

        )

        tmax = st.number_input(

            "Maximum Temperature",

            20.0,

            45.0,

            32.0

        )

        tmin = st.number_input(

            "Minimum Temperature",

            10.0,

            35.0,

            24.0

        )

        humidity = st.slider(

            "Humidity",

            20,

            100,

            80

        )

    ##########################################################

    with c2:

        pressure = st.number_input(

            "Pressure",

            980.0,

            1035.0,

            1008.0

        )

        wind = st.number_input(

            "Wind",

            0.0,

            15.0,

            2.5

        )

        solar = st.number_input(

            "Solar Radiation",

            0.0,

            12.0,

            7.0

        )

        enso = st.slider(

            "ENSO",

            -3.0,

            3.0,

            0.0

        )

    ##########################################################

    with c3:

        iod = st.slider(

            "IOD",

            -3.0,

            3.0,

            0.0

        )

        ndvi = st.slider(

            "NDVI",

            0.0,

            1.0,

            0.70

        )

        spi = st.slider(

            "SPI",

            -3.0,

            3.0,

            0.0

        )

    ##########################################################
    # DATAFRAME
    ##########################################################

    X = pd.DataFrame(

        {

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

        }

    )

    ##########################################################
    # BUTTON
    ##########################################################

    predict = st.button(

        "🚀 Predict Extreme Climate"

    )

    ##########################################################
    # PREDICTION
    ##########################################################

    if predict:

        pred = model.predict(X)

        prob = model.predict_proba(X)

        label = encoder.inverse_transform(pred)[0]

        ##############################################################
# CLIMATE RISK INDEX
##############################################################

risk = calculate_climate_risk(

    rainfall,

    tmax,

    humidity,

    spi,

    enso,

    iod

)

st.markdown("---")

        st.subheader("Prediction Result")

        st.success(

            f"Predicted Class : **{label}**"

        )

        ######################################################
        # Probability
        ######################################################

        probability = pd.DataFrame(

            {

                "Class":encoder.classes_,

                "Probability":prob[0]

            }

        )

        st.dataframe(

            probability,

            use_container_width=True

        )

        ######################################################
        # Highest Probability
        ######################################################

        best = probability.loc[

            probability["Probability"].idxmax()

        ]

        st.metric(

            "Confidence",

            f"{best['Probability']*100:.2f}%"

        )

######################################################
# GAUGE CONFIDENCE
######################################################

import plotly.graph_objects as go
import plotly.express as px

confidence = best["Probability"] * 100

gauge = go.Figure(

    go.Indicator(

        mode="gauge+number",

        value=confidence,

        title={"text":"Prediction Confidence"},

        gauge={

            "axis":{"range":[0,100]},

            "bar":{"color":"royalblue"},

            "steps":[

                {"range":[0,40],"color":"#ff4d4d"},

                {"range":[40,70],"color":"orange"},

                {"range":[70,90],"color":"yellow"},

                {"range":[90,100],"color":"green"}

            ]

        }

    )

)

gauge.update_layout(

    height=350

)

st.plotly_chart(

    gauge,

    use_container_width=True

)

######################################################
# PROBABILITY BAR CHART
######################################################

st.subheader("Prediction Probability")

bar = px.bar(

    probability,

    x="Class",

    y="Probability",

    color="Probability",

    text="Probability",

    color_continuous_scale="Turbo"

)

bar.update_traces(

    texttemplate="%{text:.2f}",

    textposition="outside"

)

bar.update_layout(

    height=450,

    yaxis_title="Probability",

    xaxis_title="Extreme Climate Class"

)

st.plotly_chart(

    bar,

    use_container_width=True

)

######################################################
# RISK LEVEL
######################################################

st.subheader("Climate Risk Level")

if label == "Flood":

    st.error("🔴 HIGH RISK : Flood Potential Detected")

elif label == "Heatwave":

    st.error("🔴 HIGH RISK : Heatwave Potential Detected")

elif label == "Drought":

    st.warning("🟠 MEDIUM RISK : Drought Potential")

else:

    st.success("🟢 LOW RISK : Normal Climate Condition")

######################################################
# MITIGATION
######################################################

st.subheader("Mitigation Recommendation")

if label=="Flood":

    st.info("""

✔ Monitor rainfall continuously

✔ Activate flood early warning

✔ Prepare evacuation route

✔ Clean drainage systems

✔ Monitor river water level

""")

elif label=="Drought":

    st.info("""

✔ Optimize water management

✔ Reduce irrigation demand

✔ Monitor groundwater

✔ Increase water storage

""")

elif label=="Heatwave":

    st.info("""

✔ Limit outdoor activities

✔ Increase hydration

✔ Protect vulnerable groups

✔ Monitor temperature continuously

""")

else:

    st.success("""

Climate condition is stable.

Continue regular monitoring.

""")

######################################################
# WEATHER STATUS CARD
######################################################

st.subheader("Current Climate Status")

col1,col2,col3,col4 = st.columns(4)

col1.metric(

    "Rainfall",

    f"{rainfall:.1f} mm"

)

col2.metric(

    "Temperature",

    f"{tmax:.1f} °C"

)

col3.metric(

    "Humidity",

    f"{humidity}%"

)

col4.metric(

    "SPI",

    f"{spi:.2f}"

)

######################################################
# DOWNLOAD RESULT
######################################################

prediction_result = pd.DataFrame({

    "Prediction":[label],

    "Confidence":[confidence],

    "Rainfall":[rainfall],

    "Temperature":[tmax],

    "Humidity":[humidity],

    "Pressure":[pressure],

    "Wind":[wind],

    "Solar":[solar],

    "ENSO":[enso],

    "IOD":[iod],

    "NDVI":[ndvi],

    "SPI":[spi]

})

csv = prediction_result.to_csv(index=False)

st.download_button(

    "📥 Download Prediction Report",

    csv,

    file_name="prediction_result.csv",

    mime="text/csv"

)

##############################################################
# EARLY WARNING GAUGE
##############################################################

st.markdown("---")

st.subheader("🚨 Climate Early Warning System")

gauge = go.Figure(

go.Indicator(

mode="gauge+number",

value=risk,

title={

"text":"Climate Risk Index"

},

gauge={

"axis":{"range":[0,100]},

"bar":{"color":"royalblue"},

"steps":[

{"range":[0,25],"color":"green"},

{"range":[25,50],"color":"yellow"},

{"range":[50,75],"color":"orange"},

{"range":[75,100],"color":"red"}

]

}

)

)

gauge.update_layout(

height=350

)

st.plotly_chart(

gauge,

use_container_width=True
)

##############################################################
# WARNING LEVEL
##############################################################

if risk >= 75:

    st.error("""

🔴 LEVEL IV

VERY HIGH RISK

Extreme climate event is very likely.

Immediate mitigation is recommended.

""")

elif risk >= 50:

    st.warning("""

🟠 LEVEL III

HIGH RISK

Preparedness should be increased.

""")

elif risk >= 25:

    st.info("""

🟡 LEVEL II

MODERATE RISK

Continuous monitoring required.

""")

else:

    st.success("""

🟢 LEVEL I

LOW RISK

Normal climate condition.

""")

##############################################################
# ALERT CARD
##############################################################

st.markdown("---")

st.subheader("📢 Alert Summary")

alert = pd.DataFrame(

{

"Indicator":[

"Rainfall",

"Temperature",

"Humidity",

"ENSO",

"IOD",

"SPI",

"Prediction",

"Risk Index"

],

"Value":[

rainfall,

tmax,

humidity,

enso,

iod,

spi,

label,

risk

]

}

)

st.dataframe(

alert,

use_container_width=True
)

##############################################################
# MITIGATION
##############################################################

st.subheader("Recommended Action")

if risk >= 75:

    st.error("""

✔ Activate Emergency Operation Center

✔ Notify Local Government

✔ Activate Early Warning System

✔ Monitor River Water Level

✔ Prepare Evacuation Route

✔ Public Information Broadcast

""")

elif risk >= 50:

    st.warning("""

✔ Increase Monitoring Frequency

✔ Coordinate with BPBD

✔ Prepare Emergency Logistics

✔ Inspect Critical Infrastructure

""")

elif risk >=25:

    st.info("""

✔ Continue Weather Monitoring

✔ Disseminate Weather Information

✔ Update Climate Dashboard

""")

else:

    st.success("""

✔ Continue Routine Monitoring

✔ No Immediate Action Required

""")

######################################################
# END
######################################################