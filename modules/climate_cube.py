##############################################################
# modules/climate_cube.py
# Indonesia Climate Data Cube (ICDC)
##############################################################

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

##############################################################
# MAIN FUNCTION
##############################################################

def climate_cube_dashboard(df):

    st.header("📦 Indonesia Climate Data Cube (ICDC)")

    st.markdown("""
Climate Data Cube merupakan representasi multidimensi data iklim
berdasarkan dimensi ruang, waktu, dan variabel iklim.
""")

    ##############################################################
    # FILTER
    ##############################################################

    years = sorted(df["Year"].unique())

    year = st.selectbox(
        "Select Year",
        years
    )

    variables = [

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

    variable = st.selectbox(

        "Climate Variable",

        variables

    )

    cube = df[df["Year"] == year]

    ##############################################################
    # KPI
    ##############################################################

    c1,c2,c3,c4 = st.columns(4)

    c1.metric(

        "Records",

        len(cube)

    )

    c2.metric(

        "Average",

        round(cube[variable].mean(),2)

    )

    c3.metric(

        "Maximum",

        round(cube[variable].max(),2)

    )

    c4.metric(

        "Minimum",

        round(cube[variable].min(),2)

    )

    ##############################################################
    # HEATMAP MONTH VS VARIABLE
    ##############################################################

    st.subheader("Monthly Climate Cube")

    pivot = cube.pivot_table(

        values=variable,

        index="Month",

        columns="Province",

        aggfunc="mean"

    )

    fig = px.imshow(

        pivot,

        text_auto=".1f",

        color_continuous_scale="Turbo",

        aspect="auto"

    )

    fig.update_layout(

        height=550,

        title=f"{variable} Distribution"

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    ##############################################################
    # 3D SCATTER
    ##############################################################

    st.subheader("3D Climate Cube")

    fig3 = px.scatter_3d(

        cube,

        x="Month",

        y="Rainfall",

        z="Humidity",

        color="Disaster",

        size="Rainfall",

        hover_name="Province"

    )

    fig3.update_layout(

        height=700

    )

    st.plotly_chart(

        fig3,

        use_container_width=True

    )

    ##############################################################
    # MONTHLY TREND
    ##############################################################

    st.subheader("Monthly Trend")

    monthly = cube.groupby(

        "Month"

    )[variable].mean().reset_index()

    line = px.line(

        monthly,

        x="Month",

        y=variable,

        markers=True,

        template="plotly_white"

    )

    line.update_layout(

        height=450

    )

    st.plotly_chart(

        line,

        use_container_width=True

    )

    ##############################################################
    # DISTRIBUTION
    ##############################################################

    st.subheader("Distribution")

    hist = px.histogram(

        cube,

        x=variable,

        nbins=20,

        color_discrete_sequence=["royalblue"]

    )

    hist.update_layout(

        height=450

    )

    st.plotly_chart(

        hist,

        use_container_width=True

    )

    ##############################################################
    # BOXPLOT
    ##############################################################

    st.subheader("Boxplot")

    box = px.box(

        cube,

        y=variable,

        color="Disaster"

    )

    st.plotly_chart(

        box,

        use_container_width=True

    )

    ##############################################################
    # SUMMARY
    ##############################################################

    st.subheader("Climate Cube Summary")

    summary = cube[variables].describe().T

    st.dataframe(

        summary,

        use_container_width=True

    )

    ##############################################################
    # DOWNLOAD
    ##############################################################

    csv = cube.to_csv(index=False)

    st.download_button(

        "📥 Download Climate Cube",

        csv,

        "Climate_Cube.csv",

        "text/csv"

    )

    ##############################################################
    # INTERPRETATION
    ##############################################################

    st.markdown("---")

    st.success(f"""

Climate Data Cube berhasil dibentuk menggunakan
dimensi:

• Space : Province

• Time : Year - Month

• Variable : {variable}

Cube ini dapat digunakan sebagai input
Machine Learning,
Deep Learning,
Explainable AI,
Decision Support System,
dan Climate Intelligence Platform.

""")