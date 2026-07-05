##############################################################
# modules/dashboard.py
# Aceh Extreme Climate Dashboard
##############################################################

import os
import streamlit as st
import plotly.express as px

from modules.maps import climate_map


##############################################################
# DASHBOARD
##############################################################

def dashboard(df):

    ##############################################################
    # FILTER
    ##############################################################

    year = st.sidebar.selectbox(

        "Year",

        sorted(df["Year"].dropna().unique())

    )

    month = st.sidebar.selectbox(

        "Month",

        sorted(df["Month"].dropna().unique())

    )

    filtered = df[

        (df["Year"] == year) &

        (df["Month"] == month)

    ]

    ##############################################################
    # HEADER
    ##############################################################

    left, center, right = st.columns([1,6,1])

    with left:

        if os.path.exists("assets/Download-PNG-Logo-Universitas-Sriwijaya_.jpg"):

            st.image(

                "assets/logo_unsri.png",

                width=100

            )

    with center:

        st.markdown("""

# 🌦 Aceh Extreme Climate Dashboard

### Climate Big Data • Machine Learning • Explainable AI

Decision Support System for Extreme Climate Analysis in Aceh

""")

    with right:

        if os.path.exists("assets/logo_bmkg.png"):

            st.image(

                "assets/logo_bmkg.png",

                width=90

            )

    ##############################################################
    # PROFILE
    ##############################################################

    col1, col2 = st.columns([4,1])

    with col1:

        st.info("""

Dashboard ini dikembangkan untuk mendukung
analisis kejadian iklim ekstrem di Provinsi Aceh.

Platform ini mengintegrasikan:

✅ Climate Big Data

✅ Machine Learning

✅ XGBoost

✅ Explainable AI (SHAP)

✅ Decision Support System

✅ Early Warning System

""")

    with col2:

        if os.path.exists("assets/Dr. Melly Ariska, S.Pd., M.Sc.jpeg"):

            st.image(

                "assets/melly.jpg",

                width=170

            )

        st.markdown("""

### Melly Ariska

Physics Education

Faculty of Teacher Training and Education

Universitas Sriwijaya

""")

    ##############################################################
    # CLIMATE SUMMARY
    ##############################################################

    st.markdown("---")

    st.subheader("🌤 Climate Summary")

    c1,c2,c3,c4,c5 = st.columns(5)

    c1.metric(

        "Rainfall",

        f"{filtered['Rainfall'].mean():.1f} mm"

    )

    c2.metric(

        "Temperature",

        f"{filtered['Tmax'].mean():.1f} °C"

    )

    c3.metric(

        "Humidity",

        f"{filtered['Humidity'].mean():.1f}%"

    )

    c4.metric(

        "Wind",

        f"{filtered['Wind'].mean():.2f} m/s"

    )

    c5.metric(

        "Pressure",

        f"{filtered['Pressure'].mean():.1f} hPa"

    )

    ##############################################################
    # MAP
    ##############################################################

    st.markdown("---")

    st.subheader("🗺 Spatial Distribution of Extreme Climate Events")

    climate_map(filtered)

    ##############################################################
    # CLIMATE DISTRIBUTION
    ##############################################################

    st.markdown("---")

    st.subheader("📊 Climate Variable Distribution")

    variable = st.selectbox(

        "Select Variable",

        [

            "Rainfall",

            "Tmax",

            "Tmin",

            "Humidity",

            "Pressure",

            "Wind",

            "Solar"

        ]

    )

    fig = px.histogram(

        filtered,

        x=variable,

        nbins=20,

        color_discrete_sequence=["royalblue"]

    )

    fig.update_layout(

        height=450,

        title=f"Distribution of {variable}"

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    ##############################################################
    # MONTHLY TIMESERIES
    ##############################################################

    st.markdown("---")

    st.subheader("📈 Climate Trend")

    variable2 = st.selectbox(

        "Trend Variable",

        [

            "Rainfall",

            "Tmax",

            "Humidity",

            "Pressure",

            "Wind"

        ],

        key="trend"

    )

    trend = (

        df

        .groupby("Month")[variable2]

        .mean()

        .reset_index()

    )

    fig2 = px.line(

        trend,

        x="Month",

        y=variable2,

        markers=True

    )

    fig2.update_layout(

        height=450

    )

    st.plotly_chart(

        fig2,

        use_container_width=True

    )

    ##############################################################
    # DATA PREVIEW
    ##############################################################

    st.markdown("---")

    st.subheader("📋 Climate Dataset")

    st.dataframe(

        filtered,

        use_container_width=True,

        hide_index=True

    )

    ##############################################################
    # FOOTER
    ##############################################################

    st.markdown("---")

    c1,c2,c3 = st.columns([1,5,1])

    with c1:

        if os.path.exists("assets/logo_unsri.png"):

            st.image(

                "assets/logo_unsri.png",

                width=60

            )

    with c2:

        st.markdown("""

### Climate Intelligence Platform

Machine Learning • Explainable AI • Decision Support System

Physics Education Department

Faculty of Teacher Training and Education

Universitas Sriwijaya

© 2026 Melly Ariska

""")

    with c3:

        if os.path.exists("assets/logo_bmkg.png"):

            st.image(

                "assets/logo_bmkg.png",

                width=60

            )
