#############################################################
# modules/correlation.py
# Correlation Analysis Dashboard
#############################################################

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px


def correlation_dashboard(df):

    st.header("🔥 Climate Variable Correlation Analysis")

    st.markdown("""
Analisis korelasi digunakan untuk mengetahui hubungan antar
variabel iklim sebelum dilakukan pemodelan Machine Learning.
""")

    #########################################################
    # Climate Variables
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

    #########################################################
    # Variable Selection
    #########################################################

    selected = st.multiselect(
        "Select Variables",
        climate_variables,
        default=climate_variables
    )

    if len(selected) < 2:
        st.warning("Please select at least two variables.")
        return

    #########################################################
    # Pastikan semua numerik
    #########################################################

    corr_df = df[selected].copy()

    for col in corr_df.columns:
        corr_df[col] = pd.to_numeric(
            corr_df[col],
            errors="coerce"
        )

    corr_df = corr_df.dropna(how="all")

    #########################################################
    # Correlation
    #########################################################

    corr = corr_df.corr(method="pearson")

    if corr.empty:
        st.error("Correlation matrix cannot be generated.")
        return

    #########################################################
    # Heatmap
    #########################################################

    fig = px.imshow(

        corr,

        text_auto=".2f",

        color_continuous_scale="RdBu_r",

        zmin=-1,

        zmax=1,

        aspect="auto"

    )

    fig.update_layout(

        title="Pearson Correlation Matrix",

        height=700

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

        corr.round(3),

        use_container_width=True

    )

    #########################################################
    # Top Correlation
    #########################################################

    corr2 = corr.abs().copy()

    # Hapus diagonal TANPA np.fill_diagonal()
    for col in corr2.columns:
        corr2.loc[col, col] = np.nan

    top = (

        corr2.stack()

        .reset_index()

    )

    top.columns = [

        "Variable 1",

        "Variable 2",

        "Correlation"

    ]

    top["Pair"] = top.apply(

        lambda x: tuple(

            sorted(

                [x["Variable 1"], x["Variable 2"]]

            )

        ),

        axis=1

    )

    top = (

        top

        .drop_duplicates(subset="Pair")

        .drop(columns="Pair")

        .sort_values(

            "Correlation",

            ascending=False

        )

    )

    st.subheader("Top Correlation")

    st.dataframe(

        top.head(10),

        use_container_width=True

    )

    #########################################################
    # Histogram
    #########################################################

    values = corr.to_numpy().flatten()

    values = values[~np.isnan(values)]

    values = values[np.abs(values) < 0.999]

    fig2 = px.histogram(

        x=values,

        nbins=25,

        color_discrete_sequence=["royalblue"]

    )

    fig2.update_layout(

        title="Distribution of Correlation",

        xaxis_title="Correlation",

        yaxis_title="Frequency",

        height=450

    )

    st.plotly_chart(

        fig2,

        use_container_width=True

    )

    #########################################################
    # Strong Correlation
    #########################################################

    st.subheader("Strong Correlation (|r| ≥ 0.70)")

    strong = []

    for i in range(len(selected)):

        for j in range(i + 1, len(selected)):

            r = corr.iloc[i, j]

            if abs(r) >= 0.70:

                strong.append([

                    selected[i],

                    selected[j],

                    round(r, 3)

                ])

    if len(strong) > 0:

        strong_df = pd.DataFrame(

            strong,

            columns=[

                "Variable A",

                "Variable B",

                "Correlation"

            ]

        )

        st.dataframe(

            strong_df,

            use_container_width=True

        )

    else:

        st.info(

            "No strong correlation detected."

        )

    #########################################################
    # Download
    #########################################################

    csv = corr.round(4).to_csv(index=True)

    st.download_button(

        "📥 Download Correlation Matrix",

        csv,

        file_name="correlation_matrix.csv",

        mime="text/csv"

    )

    #########################################################
    # Interpretation
    #########################################################

    st.markdown("---")

    st.success("""

### Interpretation

- +1 = Perfect positive correlation

- -1 = Perfect negative correlation

- 0 = Weak correlation

- |r| ≥ 0.70 = Strong correlation

- Correlation can be used as feature selection before Machine Learning.

""")
