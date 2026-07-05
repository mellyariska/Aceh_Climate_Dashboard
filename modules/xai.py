##############################################################
# modules/xai.py
# Explainable Artificial Intelligence
# Compatible with SHAP 0.47+
##############################################################

import joblib
import shap
import numpy as np
import pandas as pd

import streamlit as st

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go


##############################################################
# MAIN FUNCTION
##############################################################

def shap_dashboard(df):

    st.header("🧠 Explainable Artificial Intelligence (SHAP)")

    st.markdown("""

Model Explainability digunakan untuk mengetahui
variabel iklim yang paling berpengaruh
terhadap prediksi XGBoost.

""")

    ##############################################################
    # LOAD MODEL
    ##############################################################

    try:

        model_file = joblib.load(

            "models/xgboost.pkl"

        )

    except:

        st.error(

            "Model xgboost.pkl belum tersedia."

        )

        return

    model = model_file["model"]
        ##############################################################
    # FEATURE
    ##############################################################

    feature_columns=[

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

    ##############################################################
    # PREPROCESSING
    ##############################################################

    X=df[feature_columns].copy()

    for col in feature_columns:

        X[col]=pd.to_numeric(

            X[col],

            errors="coerce"

        )

    X=X.fillna(

        X.mean(

            numeric_only=True

        )

    )

    st.success(

        f"{len(X)} observasi berhasil dimuat."

    )
        ##############################################################
    # SHAP EXPLAINER
    ##############################################################

    st.markdown("---")
    st.subheader("⚙ SHAP Explainer")

    with st.spinner("Calculating SHAP values..."):

        try:

            ##########################################################
            # SHAP versi terbaru
            ##########################################################

            explainer = shap.Explainer(model)

            shap_values = explainer(X)

            values = shap_values.values

        except Exception:

            ##########################################################
            # Fallback jika menggunakan TreeExplainer
            ##########################################################

            explainer = shap.TreeExplainer(model)

            values = explainer.shap_values(X)

            # Jika hasil berupa list (multiclass lama)
            if isinstance(values, list):

                values = np.stack(values, axis=-1)

            # Ubah menjadi objek Explanation agar kompatibel
            shap_values = shap.Explanation(

                values=values,

                base_values=np.mean(explainer.expected_value),

                data=X.values,

                feature_names=feature_columns

            )

    ##############################################################
    # CEK DIMENSI SHAP
    ##############################################################

    st.success("SHAP values berhasil dihitung.")

    st.info(

        f"""
Jumlah Observasi : {X.shape[0]}

Jumlah Fitur : {X.shape[1]}

Dimensi SHAP : {values.shape}
"""

    )

    ##############################################################
    # NORMALISASI DIMENSI
    ##############################################################

    # Binary Classification
    if values.ndim == 2:

        shap_matrix = values

    # Multi-class
    elif values.ndim == 3:

        # rata-rata absolut seluruh kelas
        shap_matrix = np.mean(

            np.abs(values),

            axis=2

        )

    else:

        st.error(

            "Format SHAP tidak dikenali."

        )

        return

    st.success(

        f"SHAP matrix siap digunakan : {shap_matrix.shape}"

    )
        ##############################################################
    # SHAP SUMMARY PLOT
    ##############################################################

    st.markdown("---")

    st.subheader("📊 SHAP Summary Plot")

    st.write("""
Summary Plot menunjukkan pengaruh setiap variabel
terhadap prediksi model secara keseluruhan.
""")

    fig_summary = plt.figure(figsize=(10,6))

    try:

        explanation = shap.Explanation(

            values=shap_matrix,

            base_values=np.zeros(X.shape[0]),

            data=X.values,

            feature_names=feature_columns

        )

        shap.plots.beeswarm(

            explanation,

            max_display=20,

            show=False

        )

        st.pyplot(

            fig_summary,

            use_container_width=True

        )

    except Exception as e:

        st.error(

            f"Summary plot gagal : {e}"

        )

    plt.close()

    ##############################################################
    # BAR SUMMARY
    ##############################################################

    st.markdown("---")

    st.subheader("📈 SHAP Summary Bar")

    fig_bar = plt.figure(figsize=(10,6))

    try:

        shap.plots.bar(

            explanation,

            max_display=20,

            show=False

        )

        st.pyplot(

            fig_bar,

            use_container_width=True

        )

    except Exception as e:

        st.error(

            f"Bar Plot gagal : {e}"

        )

    plt.close()
        ##############################################################
    # GLOBAL FEATURE IMPORTANCE
    ##############################################################

    st.markdown("---")
    st.subheader("🌍 Global SHAP Feature Importance")

    ##############################################################
    # Mean Absolute SHAP
    ##############################################################

    mean_shap = np.mean(

        np.abs(shap_matrix),

        axis=0

    )

    ##############################################################
    # DataFrame
    ##############################################################

    importance = pd.DataFrame(

        {

            "Feature": feature_columns,

            "Mean_SHAP": mean_shap

        }

    )

    importance = importance.sort_values(

        by="Mean_SHAP",

        ascending=False

    ).reset_index(drop=True)

    ##############################################################
    # TABLE
    ##############################################################

    st.dataframe(

        importance,

        use_container_width=True,

        hide_index=True

    )

    ##############################################################
    # HORIZONTAL BAR
    ##############################################################

    fig = px.bar(

        importance,

        x="Mean_SHAP",

        y="Feature",

        orientation="h",

        color="Mean_SHAP",

        color_continuous_scale="Viridis",

        text_auto=".4f"

    )

    fig.update_layout(

        title="Global SHAP Feature Importance",

        height=600,

        yaxis=dict(categoryorder="total ascending")

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    ##############################################################
    # PIE CHART
    ##############################################################

    st.markdown("---")

    st.subheader("🥧 SHAP Contribution")

    fig2 = px.pie(

        importance.head(10),

        names="Feature",

        values="Mean_SHAP",

        hole=0.45,

        color_discrete_sequence=px.colors.qualitative.Set3

    )

    fig2.update_layout(

        height=500

    )

    st.plotly_chart(

        fig2,

        use_container_width=True

    )

    ##############################################################
    # TOP FEATURE
    ##############################################################

    st.success(

        f"""
Most influential feature :

**{importance.iloc[0]['Feature']}**

Mean SHAP Value :

**{importance.iloc[0]['Mean_SHAP']:.5f}**
"""

    )
        ##############################################################
    # LOCAL EXPLANATION
    ##############################################################

    st.markdown("---")
    st.subheader("🔍 Local Explainability")

    st.write("""
Pilih satu observasi untuk melihat kontribusi masing-masing
variabel iklim terhadap hasil prediksi model.
""")

    ##############################################################
    # SELECT OBSERVATION
    ##############################################################

    idx = st.slider(

        "Observation Index",

        min_value=0,

        max_value=len(X)-1,

        value=0,

        step=1

    )

    ##############################################################
    # DATA OBSERVATION
    ##############################################################

    st.markdown("### 📋 Selected Climate Data")

    st.dataframe(

        X.iloc[[idx]],

        use_container_width=True,

        hide_index=True

    )

    ##############################################################
    # PREDICTION
    ##############################################################

    try:

        prediction = model.predict(

            X.iloc[[idx]]

        )[0]

        probability = model.predict_proba(

            X.iloc[[idx]]

        )[0]

        predicted_label = model_file["encoder"].inverse_transform(

            [prediction]

        )[0]

        st.success(

            f"Predicted Class : **{predicted_label}**"

        )

    except Exception:

        st.info(

            "Model tidak memiliki encoder."

        )

        probability = None

    ##############################################################
    # PROBABILITY
    ##############################################################

    if probability is not None:

        prob_df = pd.DataFrame(

            {

                "Class":

                    model_file["encoder"].classes_,

                "Probability":

                    probability

            }

        )

        fig_prob = px.bar(

            prob_df,

            x="Class",

            y="Probability",

            color="Probability",

            color_continuous_scale="Viridis",

            text_auto=".2f"

        )

        fig_prob.update_layout(

            height=400,

            title="Prediction Probability"

        )

        st.plotly_chart(

            fig_prob,

            use_container_width=True

        )

    ##############################################################
    # WATERFALL PLOT
    ##############################################################

    st.markdown("---")

    st.subheader("💧 SHAP Waterfall Plot")

    try:

        fig_water = plt.figure(figsize=(10,7))

        shap.plots.waterfall(

            shap_values[idx],

            max_display=15,

            show=False

        )

        st.pyplot(

            fig_water,

            use_container_width=True

        )

        plt.close()

    except Exception as e:

        st.warning(

            f"Waterfall Plot tidak tersedia : {e}"

        )

    ##############################################################
    # LOCAL FEATURE CONTRIBUTION
    ##############################################################

    st.markdown("---")

    st.subheader("📈 Local Feature Contribution")

    local_df = pd.DataFrame(

        {

            "Feature":

                feature_columns,

            "Contribution":

                shap_matrix[idx]

        }

    )

    local_df = local_df.sort_values(

        "Contribution",

        key=np.abs,

        ascending=False

    )

    st.dataframe(

        local_df,

        use_container_width=True,

        hide_index=True

    )

    ##############################################################
    # BAR CHART
    ##############################################################

    fig_local = px.bar(

        local_df,

        x="Contribution",

        y="Feature",

        orientation="h",

        color="Contribution",

        color_continuous_scale="RdBu"

    )

    fig_local.update_layout(

        height=600,

        title="Local SHAP Contribution"

    )

    st.plotly_chart(

        fig_local,

        use_container_width=True

    )
        ##############################################################
    # SHAP DEPENDENCE PLOT
    ##############################################################

    st.markdown("---")
    st.subheader("📉 SHAP Dependence Plot")

    selected_feature = st.selectbox(

        "Select Feature",

        feature_columns,

        index=0

    )

    feature_index = feature_columns.index(selected_feature)

    ##############################################################
    # DEPENDENCE SCATTER
    ##############################################################

    dep_df = pd.DataFrame({

        "Feature Value": X[selected_feature],

        "SHAP Value": shap_matrix[:, feature_index]

    })

    fig_dep = px.scatter(

        dep_df,

        x="Feature Value",

        y="SHAP Value",

        color="SHAP Value",

        color_continuous_scale="Viridis",

        trendline="ols",

        opacity=0.75

    )

    fig_dep.update_layout(

        title=f"SHAP Dependence Plot - {selected_feature}",

        height=550

    )

    st.plotly_chart(

        fig_dep,

        use_container_width=True

    )

    ##############################################################
    # FEATURE DISTRIBUTION
    ##############################################################

    st.markdown("---")

    st.subheader("📊 Feature Distribution")

    fig_hist = px.histogram(

        X,

        x=selected_feature,

        nbins=30,

        color_discrete_sequence=["royalblue"]

    )

    fig_hist.update_layout(

        height=420

    )

    st.plotly_chart(

        fig_hist,

        use_container_width=True

    )

    ##############################################################
    # FEATURE STATISTICS
    ##############################################################

    st.markdown("---")

    st.subheader("📋 Feature Statistics")

    stat = pd.DataFrame({

        "Statistic":[

            "Mean",

            "Median",

            "Minimum",

            "Maximum",

            "Standard Deviation"

        ],

        "Value":[

            X[selected_feature].mean(),

            X[selected_feature].median(),

            X[selected_feature].min(),

            X[selected_feature].max(),

            X[selected_feature].std()

        ]

    })

    st.dataframe(

        stat,

        use_container_width=True,

        hide_index=True

    )

    ##############################################################
    # CORRELATION WITH SHAP
    ##############################################################

    st.markdown("---")

    st.subheader("📈 Correlation Between Feature and SHAP")

    corr = np.corrcoef(

        X[selected_feature],

        shap_matrix[:, feature_index]

    )[0,1]

    c1, c2 = st.columns(2)

    c1.metric(

        "Correlation",

        f"{corr:.3f}"

    )

    c2.metric(

        "Mean |SHAP|",

        f"{np.abs(shap_matrix[:, feature_index]).mean():.4f}"

    )

    ##############################################################
    # SENSITIVITY ANALYSIS
    ##############################################################

    st.markdown("---")

    st.subheader("🎯 Sensitivity Analysis")

    sensitivity = pd.DataFrame({

        "Feature Value": X[selected_feature],

        "Absolute SHAP": np.abs(

            shap_matrix[:, feature_index]

        )

    })

    fig_sens = px.scatter(

        sensitivity,

        x="Feature Value",

        y="Absolute SHAP",

        color="Absolute SHAP",

        color_continuous_scale="Turbo",

        opacity=0.75

    )

    fig_sens.update_layout(

        height=500,

        title=f"Sensitivity of {selected_feature}"

    )

    st.plotly_chart(

        fig_sens,

        use_container_width=True

    )

    ##############################################################
    # INTERPRETATION
    ##############################################################

    st.markdown("---")

    st.success(f"""

### 🔎 Interpretation

Selected Feature :

**{selected_feature}**

Correlation between feature value and SHAP value :

**{corr:.3f}**

Average absolute SHAP contribution :

**{np.abs(shap_matrix[:, feature_index]).mean():.5f}**

Semakin besar nilai absolut SHAP,
semakin besar kontribusi variabel tersebut
terhadap prediksi model.

""")
        ##############################################################
    # DOWNLOAD SHAP FEATURE IMPORTANCE
    ##############################################################

    st.markdown("---")

    st.subheader("📥 Download Results")

    csv_importance = importance.to_csv(index=False)

    st.download_button(

        label="📥 Download SHAP Feature Importance",

        data=csv_importance,

        file_name="shap_feature_importance.csv",

        mime="text/csv"

    )

    ##############################################################
    # DOWNLOAD LOCAL EXPLANATION
    ##############################################################

    csv_local = local_df.to_csv(index=False)

    st.download_button(

        label="📥 Download Local Explanation",

        data=csv_local,

        file_name=f"local_explanation_{idx}.csv",

        mime="text/csv"

    )

    ##############################################################
    # TOP 5 FEATURES
    ##############################################################

    st.markdown("---")

    st.subheader("🏆 Top 5 Most Influential Variables")

    top5 = importance.head(5)

    st.dataframe(

        top5,

        use_container_width=True,

        hide_index=True

    )

    ##############################################################
    # EXPLAINABILITY SCORE
    ##############################################################

    explainability_score = float(

        importance["Mean_SHAP"].mean()

    )

    st.markdown("---")

    st.subheader("🎯 Explainability Score")

    gauge = go.Figure(

        go.Indicator(

            mode="gauge+number",

            value=explainability_score*100,

            title={"text":"Explainability Score"},

            gauge={

                "axis":{"range":[0,100]},

                "bar":{"color":"royalblue"},

                "steps":[

                    {"range":[0,20],"color":"#ffcccc"},

                    {"range":[20,40],"color":"#ffe599"},

                    {"range":[40,60],"color":"#fff2cc"},

                    {"range":[60,80],"color":"#b6d7a8"},

                    {"range":[80,100],"color":"#6aa84f"}

                ]

            }

        )

    )

    gauge.update_layout(

        height=420

    )

    st.plotly_chart(

        gauge,

        use_container_width=True

    )

    ##############################################################
    # SUMMARY DASHBOARD
    ##############################################################

    st.markdown("---")

    st.subheader("📊 Explainable AI Summary")

    c1,c2,c3 = st.columns(3)

    c1.metric(

        "Observations",

        len(X)

    )

    c2.metric(

        "Features",

        len(feature_columns)

    )

    c3.metric(

        "Mean SHAP",

        f"{importance['Mean_SHAP'].mean():.5f}"

    )

    ##############################################################
    # INTERPRETATION
    ##############################################################

    st.success(f"""

### 🧠 Explainable AI Report

Model berhasil dijelaskan menggunakan **SHAP (SHapley Additive Explanations)**.

**Jumlah Observasi :**
{len(X)}

**Jumlah Variabel :**
{len(feature_columns)}

**Top Feature :**
{importance.iloc[0]['Feature']}

**Mean SHAP :**
{importance.iloc[0]['Mean_SHAP']:.5f}

**Explainability Score :**
{explainability_score:.5f}

Interpretasi:

• Nilai SHAP positif meningkatkan peluang kelas yang diprediksi.

• Nilai SHAP negatif menurunkan peluang kelas yang diprediksi.

• Semakin besar |SHAP| maka semakin besar kontribusi variabel terhadap keputusan model.

Dashboard ini kompatibel dengan:

✅ XGBoost 2.x

✅ SHAP 0.47+

✅ Streamlit Cloud

""")

    ##############################################################
    # EXPORT COMPLETE REPORT
    ##############################################################

    report = pd.concat(

        [

            importance,

            pd.DataFrame({

                "Feature":["Explainability Score"],

                "Mean_SHAP":[explainability_score]

            })

        ],

        ignore_index=True

    )

    st.download_button(

        "📄 Export Explainable AI Report",

        report.to_csv(index=False),

        file_name="Explainable_AI_Report.csv",

        mime="text/csv"

    )
