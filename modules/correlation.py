#############################################################
# modules/correlation.py
# Correlation Analysis Module
#############################################################

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

#############################################################
# FUNCTION
#############################################################

def correlation_dashboard(df):

    st.header("🔥 Climate Variable Correlation Analysis")

    st.markdown(
        """
        Analisis korelasi digunakan untuk mengetahui hubungan antar
        variabel iklim yang digunakan dalam pemodelan Machine Learning.
        """
    )

    #########################################################
    # Variable Selection
    #########################################################

    climate_variables = [

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

    selected = st.multiselect(

        "Select Variables",

        climate_variables,

        default=climate_variables

    )

    #########################################################
    # Correlation Matrix
    #########################################################

    corr = df[selected].corr(method="pearson")

    #########################################################
    # Heatmap
    #########################################################

    fig = px.imshow(

        corr,

        text_auto=".2f",

        color_continuous_scale="RdBu_r",

        aspect="auto",

        origin="lower"

    )

    fig.update_layout(

        title="Pearson Correlation Matrix",

        height=700,

        font=dict(

            size=14

        )

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    #########################################################
    # Correlation Table
    #########################################################

    st.subheader("Correlation Matrix")

    st.dataframe(

        corr.style.background_gradient(

            cmap="RdYlBu"

        )

    )

    #########################################################
    # Highest Correlation
    #########################################################

    corr2 = corr.abs()

    corr2.values[
        range(len(corr2)),
        range(len(corr2))
    ] = 0

    max_corr = corr2.unstack().sort_values(
        ascending=False
    )

    st.subheader("Top Correlation")

    top = max_corr.reset_index()

    top.columns = [

        "Variable 1",

        "Variable 2",

        "Correlation"

    ]

    top = top.drop_duplicates(
        subset=["Correlation"]
    )

    st.dataframe(

        top.head(10)

    )

    #########################################################
    # Histogram Correlation
    #########################################################

    st.subheader("Distribution of Correlation")

    values = corr.values.flatten()

    values = values[values != 1]

    hist = px.histogram(

        x=values,

        nbins=30,

        color_discrete_sequence=["royalblue"]

    )

    hist.update_layout(

        xaxis_title="Correlation",

        yaxis_title="Frequency",

        height=450

    )

    st.plotly_chart(

        hist,

        use_container_width=True

    )

    #########################################################
    # Strong Correlation
    #########################################################

    st.subheader("Strong Relationship")

    strong = []

    for i in range(len(selected)):

        for j in range(i+1,len(selected)):

            r = corr.iloc[i,j]

            if abs(r)>=0.70:

                strong.append([

                    selected[i],

                    selected[j],

                    round(r,3)

                ])

    if len(strong)>0:

        strong_df = pd.DataFrame(

            strong,

            columns=[

                "Variable A",

                "Variable B",

                "Correlation"

            ]

        )

        st.dataframe(

            strong_df

        )

    else:

        st.info("No strong correlation detected.")

    #########################################################
    # Download Matrix
    #########################################################

    csv = corr.to_csv()

    st.download_button(

        label="📥 Download Correlation Matrix",

        data=csv,

        file_name="correlation_matrix.csv",

        mime="text/csv"

    )

    #########################################################
    # Interpretation
    #########################################################

    st.markdown("---")

    st.success("""

Interpretasi Korelasi

• Nilai +1 menunjukkan hubungan positif sempurna.

• Nilai -1 menunjukkan hubungan negatif sempurna.

• Nilai mendekati 0 menunjukkan hubungan lemah.

• Korelasi >0.70 dianggap kuat.

• Korelasi < -0.70 dianggap negatif kuat.

Nilai korelasi ini dapat digunakan
sebagai dasar Feature Selection
sebelum pemodelan Machine Learning.

""")