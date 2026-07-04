##############################################################
# ACEH EXTREME CLIMATE DASHBOARD
# Integrating Multi-Source Climate Big Data
# Author : Melly Ariska
##############################################################

import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

from PIL import Image
from modules.maps import climate_map
from modules.correlation import correlation_dashboard
from modules.cdrs import cdrs_dashboard
from modules.climate_cube import climate_cube_dashboard
from modules.random_forest import random_forest_dashboard
from modules.xai import shap_dashboard

##############################################################

# MAP
###########################################################

if menu=="Dashboard":

    climate_map(filtered)

###########################################################

# PAGE CONFIG
##############################################################

st.set_page_config(
    page_title="Aceh Extreme Climate Dashboard",
    page_icon="🌦",
    layout="wide",
    initial_sidebar_state="expanded"
)

##############################################################
# CUSTOM CSS
##############################################################

st.markdown("""
<style>

.main{
background-color:#f6f8fb;
}

.metric-container{
background:white;
padding:10px;
border-radius:10px;
box-shadow:2px 2px 10px lightgrey;
}

</style>

""",unsafe_allow_html=True)

##############################################################
# HEADER
##############################################################

col1,col2=st.columns([1,8])

with col1:

    st.image(
        "https://cdn-icons-png.flaticon.com/512/414/414825.png",
        width=70
    )

with col2:

    st.title("ACEH EXTREME CLIMATE DASHBOARD")

    st.caption(
        "Climate Big Data | Machine Learning | Explainable AI"
    )

st.markdown("---")

##############################################################
# LOAD DATA
##############################################################

@st.cache_data

def load_data():

    df=pd.read_excel(
        "data_aceh.xlsx"
    )

    df.columns=df.columns.str.strip()

    return df

df=load_data()

##############################################################
# SIDEBAR
##############################################################

st.sidebar.image(

"https://cdn-icons-png.flaticon.com/512/414/414825.png",

width=120

)

st.sidebar.title("Navigation")

menu = st.sidebar.radio(
    "Menu",
    [
        "Dashboard",
        "Analytics",
        "CDRS",
        "Machine Learning",
        "Prediction",
        "Climate Data Cube",
        "Report"
    ]
)

if menu == "CDRS":
    cdrs_dashboard(df)

if menu == "Analytics":

    correlation_dashboard(df)

if menu == "Climate Data Cube":

    climate_cube_dashboard(df)

if menu == "Machine Learning":
    random_forest_dashboard(df)

if menu=="Explainable AI":

    shap_dashboard(df)

st.sidebar.markdown("---")

year=st.sidebar.selectbox(

"Year",

sorted(df.Year.unique())

)

month=st.sidebar.selectbox(

"Month",

sorted(df.Month.unique())

)

filtered=df[

(df.Year==year)

&

(df.Month==month)

]

##############################################################
# KPI
##############################################################

st.subheader("Climate Summary")

c1,c2,c3,c4,c5=st.columns(5)

with c1:

    st.metric(

        "Rainfall",

        f"{filtered.Rainfall.mean():.1f} mm"

    )

with c2:

    st.metric(

        "Temperature",

        f"{filtered.Tmax.mean():.1f} °C"

    )

with c3:

    st.metric(

        "Humidity",

        f"{filtered.Humidity.mean():.1f}%"

    )

with c4:

    st.metric(

        "Wind",

        f"{filtered.Wind.mean():.2f} m/s"

    )

with c5:

    st.metric(

        "Pressure",

        f"{filtered.Pressure.mean():.1f} hPa"

    )

st.markdown("---")

##############################################################
# RAINFALL
##############################################################

st.subheader("Rainfall Trend")

fig=px.line(

df,

x="Year",

y="Rainfall",

markers=True,

color="Disaster",

template="plotly_white"

)

fig.update_layout(

height=500,

title="Rainfall Time Series"

)

st.plotly_chart(

fig,

use_container_width=True

)

##############################################################
# TEMPERATURE
##############################################################

st.subheader("Temperature")

fig=px.line(

df,

x="Year",

y="Tmax",

color="Disaster",

template="plotly_white"

)

st.plotly_chart(

fig,

use_container_width=True

)

##############################################################
# HUMIDITY
##############################################################

fig=px.area(

df,

x="Year",

y="Humidity",

template="plotly_white"

)

st.plotly_chart(

fig,

use_container_width=True

)

##############################################################
# ENSO IOD
##############################################################

fig=go.Figure()

fig.add_trace(

go.Scatter(

x=df.Year,

y=df.ENSO,

name="ENSO"

)

)

fig.add_trace(

go.Scatter(

x=df.Year,

y=df.IOD,

name="IOD"

)

)

st.plotly_chart(

fig,

use_container_width=True

)

##############################################################
# DISASTER
##############################################################

fig=px.histogram(

df,

x="Disaster",

color="Disaster"

)

st.plotly_chart(

fig,

use_container_width=True

)

##############################################################
# FOOTER
##############################################################

st.markdown("---")

st.caption("""

Aceh Extreme Climate Dashboard

Developed using

Streamlit

Plotly

Machine Learning

Climate Big Data

""")

###########################################################
# LOAD CSS
###########################################################

def local_css(file_name):

    with open(file_name) as f:

        st.markdown(

            f"<style>{f.read()}</style>",

            unsafe_allow_html=True

        )

local_css("css/style.css")

###########################################################
# HEADER
###########################################################

c1,c2,c3=st.columns([1,6,1])

with c1:

    st.image(

        "assets/logo_unsri.png",

        width=80

    )

with c2:

    st.markdown("""

# 🌦 Climate Intelligence Platform

### Machine Learning • Explainable AI • Decision Support System

Indonesia Climate Big Data Integration Framework (CBDIF)

""")

with c3:

    st.image(

        "assets/logo_bmkg.png",

        width=80

    )
st.markdown("---")

st.markdown("""

<div class="footer">

Climate Intelligence Platform Version 1.0

Developed by

Melly Ariska

Physics Education

Universitas Sriwijaya

2026

</div>

""",

unsafe_allow_html=True)

menu=option_menu(

menu_title=None,

options=[

"Dashboard",

"Analytics",

"Correlation",

"CDRS",

"Climate Cube",

"Random Forest",

"XGBoost",

"Explainable AI",

"Prediction",

"Report"

],

icons=[

"house",

"graph-up",

"grid",

"bullseye",

"database",

"tree",

"cpu",

"robot",

"activity",

"file-earmark"

],

orientation="horizontal"

)

if menu=="Dashboard":

    dashboard(df)

elif menu=="Analytics":

    analytics(df)

elif menu=="Correlation":

    correlation_dashboard(df)

elif menu=="CDRS":

    cdrs_dashboard(df)

elif menu=="Climate Cube":

    climate_cube_dashboard(df)

elif menu=="Random Forest":

    random_forest_dashboard(df)

elif menu=="XGBoost":

    xgboost_dashboard(df)

elif menu=="Explainable AI":

    shap_dashboard(df)

elif menu=="Prediction":

    prediction_dashboard()

elif menu=="Report":

    report_dashboard()
