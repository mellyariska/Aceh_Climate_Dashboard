##############################################################
# Aceh Extreme Climate Dashboard
# app.py
##############################################################

import streamlit as st
import pandas as pd

##############################################################
# PAGE CONFIG
##############################################################

st.set_page_config(

    page_title="Aceh Extreme Climate Dashboard",

    page_icon="🌦",

    layout="wide"

)

##############################################################
# LOAD CSS
##############################################################

def load_css():

    try:

        with open("css/style.css", encoding="utf-8") as f:

            st.markdown(

                f"<style>{f.read()}</style>",

                unsafe_allow_html=True

            )

    except:

        pass


load_css()

##############################################################
# LOAD DATA
##############################################################

@st.cache_data

def load_data():

    files=[

        "data/data_aceh_1985_2025.xlsx",

        "data_aceh_1985_2025.xlsx",

        "data/data_aceh.xlsx",

        "data_aceh.xlsx"

    ]

    for file in files:

        try:

            df=pd.read_excel(file)

            df.columns=df.columns.str.strip()

            return df

        except:

            continue

    st.error("Dataset tidak ditemukan.")

    st.stop()


df=load_data()

##############################################################
# IMPORT MODULE
##############################################################

from modules.dashboard import dashboard
from modules.correlation import correlation_dashboard
from modules.cdrs import cdrs_dashboard
from modules.climate_cube import climate_cube_dashboard
from modules.random_forest import random_forest_dashboard
from modules.xgboost import xgboost_dashboard
from modules.xai import shap_dashboard
from modules.prediction_dashboard import prediction_dashboard
from modules.report import report_dashboard

##############################################################
# SIDEBAR
##############################################################

st.sidebar.title("Filter Data")

menu=st.sidebar.radio(

    "Menu",

    [

        "Dashboard",

        "Correlation",

        "CDRS",

        "Climate Cube",

        "Random Forest",

        "XGBoost",

        "Explainable AI",

        "Prediction",

        "Report"

    ]

)

##############################################################
# ROUTING
##############################################################

if menu=="Dashboard":

    dashboard(df)

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
