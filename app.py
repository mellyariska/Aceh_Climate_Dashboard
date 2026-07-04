
import streamlit as st
import pandas as pd

from modules.maps import climate_map
from modules.correlation import correlation_dashboard
from modules.cdrs import cdrs_dashboard
from modules.climate_cube import climate_cube_dashboard
from modules.random_forest import random_forest_dashboard
from modules.xai import shap_dashboard

try:
    from modules.xgboost import xgboost_dashboard
except:
    xgboost_dashboard=None

try:
    from modules.prediction_dashboard import prediction_dashboard
except:
    prediction_dashboard=None

try:
    from modules.report import report_dashboard
except:
    report_dashboard=None

st.set_page_config(
    page_title="Aceh Extreme Climate Dashboard",
    page_icon="🌦",
    layout="wide"
)

try:
    with open("css/style.css","r",encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except:
    pass

st.title("🌦 Aceh Extreme Climate Dashboard")
st.caption("Climate Big Data • Machine Learning • Explainable AI")

@st.cache_data
def load_data():
    files=[
        "data_aceh_1985_2025.xlsx",
        "data/data_aceh_1985_2025.xlsx",
        "data/data_aceh_1985_2025.xlsx"
    ]
    for f in files:
        try:
            df=pd.read_excel(f)
            df.columns=df.columns.str.strip()
            return df
        except:
            pass
    st.error("Dataset tidak ditemukan.")
    st.stop()

df=load_data()

st.sidebar.header("Filter Data")

year=st.sidebar.selectbox("Year", sorted(df["Year"].unique()))
month=st.sidebar.selectbox("Month", sorted(df["Month"].unique()))

filtered=df[(df["Year"]==year)&(df["Month"]==month)]

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

if menu=="Dashboard":

    st.subheader("Climate Summary")

    c1,c2,c3,c4,c5=st.columns(5)

    c1.metric("Rainfall",f"{filtered['Rainfall'].mean():.1f} mm")
    c2.metric("Temperature",f"{filtered['Tmax'].mean():.1f} °C")
    c3.metric("Humidity",f"{filtered['Humidity'].mean():.1f}%")
    c4.metric("Wind",f"{filtered['Wind'].mean():.2f} m/s")
    c5.metric("Pressure",f"{filtered['Pressure'].mean():.1f} hPa")

    climate_map(filtered)

elif menu=="Correlation":
    correlation_dashboard(df)

elif menu=="CDRS":
    cdrs_dashboard(df)

elif menu=="Climate Cube":
    climate_cube_dashboard(df)

elif menu=="Random Forest":
    random_forest_dashboard(df)

elif menu=="XGBoost":
    if xgboost_dashboard:
        xgboost_dashboard(df)
    else:
        st.warning("Module xgboost.py belum tersedia.")

elif menu=="Explainable AI":
    shap_dashboard(df)

elif menu=="Prediction":
    if prediction_dashboard:
        prediction_dashboard()
    else:
        st.warning("prediction_dashboard.py belum tersedia.")

elif menu=="Report":
    if report_dashboard:
        report_dashboard()
    else:
        st.warning("report.py belum memiliki report_dashboard().")

st.markdown("---")
st.caption("Developed by Melly Ariska | Universitas Sriwijaya")
