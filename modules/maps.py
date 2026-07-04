###########################################################
# MAPS.PY
# Interactive Climate Map
###########################################################

import streamlit as st
import folium

from streamlit_folium import st_folium
from folium.plugins import HeatMap

###########################################################
# FUNCTION
###########################################################

def climate_map(df):

    st.subheader("🗺 Spatial Distribution of Extreme Climate Events")

    # Titik tengah Aceh
    aceh = [4.695135, 96.749399]

    # Membuat peta
    m = folium.Map(

        location=aceh,

        zoom_start=7,

        tiles="CartoDB positron"

    )

    ###########################################################
    # Marker berdasarkan jenis bencana
    ###########################################################

    for _, row in df.iterrows():

        if row["Disaster"] == "Flood":

            color = "blue"
            icon = "tint"

        elif row["Disaster"] == "Drought":

            color = "orange"
            icon = "fire"

        elif row["Disaster"] == "Heatwave":

            color = "red"
            icon = "sun"

        else:

            color = "green"
            icon = "cloud"

        folium.Marker(

            location=[

                row["Latitude"],
                row["Longitude"]

            ],

            popup=f"""
            <b>Province</b> : {row['Province']}<br>
            <b>Rainfall</b> : {row['Rainfall']} mm<br>
            <b>Tmax</b> : {row['Tmax']} °C<br>
            <b>Humidity</b> : {row['Humidity']} %<br>
            <b>Disaster</b> : {row['Disaster']}
            """,

            tooltip=row["Province"],

            icon=folium.Icon(

                color=color,

                icon=icon,

                prefix="fa"

            )

        ).add_to(m)

    ###########################################################
    # HeatMap
    ###########################################################

    heat = []

    for _, row in df.iterrows():

        heat.append([

            row["Latitude"],
            row["Longitude"],
            row["Rainfall"]

        ])

    HeatMap(

        heat,

        radius=18,

        blur=12,

        min_opacity=0.3

    ).add_to(m)

    ###########################################################
    # Tampilkan Peta
    ###########################################################

    st_folium(

        m,

        width=1200,

        height=650,

        returned_objects=[]

    )
