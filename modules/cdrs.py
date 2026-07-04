##############################################################
# modules/cdrs.py
# Climate Data Readiness Score (CDRS)
##############################################################

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


##############################################################
# FUNCTION
##############################################################

def calculate_cdrs(df):

    ##########################################################
    # Completeness
    ##########################################################

    completeness = (
        (1 - df.isnull().sum().sum() /
         (df.shape[0] * df.shape[1])) * 100
    )

    ##########################################################
    # Consistency
    ##########################################################

    duplicate = df.duplicated().sum()

    consistency = (
        (1 - duplicate / len(df))
    ) * 100

    ##########################################################
    # Spatial Alignment
    ##########################################################

    if ("Latitude" in df.columns) and ("Longitude" in df.columns):

        spatial = 100

    else:

        spatial = 70

    ##########################################################
    # Temporal Alignment
    ##########################################################

    if ("Year" in df.columns) and ("Month" in df.columns):

        temporal = 100

    else:

        temporal = 70

    ##########################################################
    # Metadata Quality
    ##########################################################

    metadata_columns = [

        "Year",
        "Month",
        "Province",
        "Latitude",
        "Longitude"

    ]

    metadata = 0

    for c in metadata_columns:

        if c in df.columns:

            metadata += 20

    ##########################################################
    # Interoperability
    ##########################################################

    interoperability = 95

    ##########################################################
    # Overall Score
    ##########################################################

    overall = np.mean([

        completeness,

        consistency,

        spatial,

        temporal,

        metadata,

        interoperability

    ])

    return {

        "Completeness": round(completeness,2),

        "Consistency": round(consistency,2),

        "Spatial Alignment": round(spatial,2),

        "Temporal Alignment": round(temporal,2),

        "Metadata Quality": round(metadata,2),

        "Interoperability": round(interoperability,2),

        "Overall": round(overall,2)

    }


##############################################################
# DASHBOARD
##############################################################

def cdrs_dashboard(df):

    st.header("📊 Climate Data Readiness Score (CDRS)")

    score = calculate_cdrs(df)

    ##########################################################
    # KPI
    ##########################################################

    c1,c2,c3 = st.columns(3)

    c1.metric(

        "Overall CDRS",

        score["Overall"]

    )

    c2.metric(

        "Completeness",

        score["Completeness"]

    )

    c3.metric(

        "Consistency",

        score["Consistency"]

    )

    ##########################################################
    # Radar Chart
    ##########################################################

    categories = [

        "Completeness",

        "Consistency",

        "Spatial Alignment",

        "Temporal Alignment",

        "Metadata Quality",

        "Interoperability"

    ]

    values = [

        score[c]

        for c in categories

    ]

    values += values[:1]

    categories += categories[:1]

    fig = go.Figure()

    fig.add_trace(

        go.Scatterpolar(

            r=values,

            theta=categories,

            fill="toself",

            line=dict(

                color="royalblue",

                width=3

            ),

            name="CDRS"

        )

    )

    fig.update_layout(

        polar=dict(

            radialaxis=dict(

                visible=True,

                range=[0,100]

            )

        ),

        showlegend=False,

        height=650,

        title="Climate Data Readiness Score"

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    ##########################################################
    # Table
    ##########################################################

    st.subheader("CDRS Summary")

    table = pd.DataFrame(

        {

            "Dimension": list(score.keys()),

            "Score": list(score.values())

        }

    )

    st.dataframe(

        table,

        use_container_width=True

    )

    ##########################################################
    # Gauge
    ##########################################################

    fig2 = go.Figure(

        go.Indicator(

            mode="gauge+number",

            value=score["Overall"],

            title={"text":"Overall CDRS"},

            gauge={

                "axis":{"range":[0,100]},

                "bar":{"color":"royalblue"},

                "steps":[

                    {"range":[0,50],"color":"red"},

                    {"range":[50,70],"color":"orange"},

                    {"range":[70,85],"color":"yellow"},

                    {"range":[85,100],"color":"green"}

                ]

            }

        )

    )

    fig2.update_layout(

        height=420

    )

    st.plotly_chart(

        fig2,

        use_container_width=True

    )

    ##########################################################
    # Interpretation
    ##########################################################

    st.markdown("---")

    if score["Overall"]>=90:

        st.success("""

Excellent Dataset

Dataset sangat siap digunakan
untuk Machine Learning.

""")

    elif score["Overall"]>=80:

        st.info("""

Good Dataset

Masih terdapat sedikit
perbaikan yang diperlukan.

""")

    elif score["Overall"]>=70:

        st.warning("""

Moderate Dataset

Disarankan melakukan
preprocessing ulang.

""")

    else:

        st.error("""

Poor Dataset

Dataset belum layak
digunakan untuk pemodelan.

""")

    ##########################################################
    # Download
    ##########################################################

    csv = table.to_csv(index=False)

    st.download_button(

        "📥 Download CDRS",

        csv,

        "CDRS_Result.csv",

        "text/csv"

    )