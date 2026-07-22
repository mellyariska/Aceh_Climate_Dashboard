# simplified xai.py
import os,joblib,shap,numpy as np,pandas as pd,streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px

def shap_dashboard(df):
    st.header("🧠 Explainable AI (SHAP)")
    MODEL_PATH="models/xgboost.pkl"
    if not os.path.exists(MODEL_PATH):
        st.error("Model XGBoost belum tersedia.")
        return
    m=joblib.load(MODEL_PATH)
    model=m["model"]; encoder=m["encoder"]; features=m["features"]
    X=m["X_train"].copy() if "X_train" in m else df[features].apply(pd.to_numeric,errors="coerce").fillna(df[features].mean(numeric_only=True))
    explainer=shap.Explainer(model)
    sv=explainer(X)
    vals=sv.values
    if vals.ndim==3:
        vals=vals[:,:,0] if vals.shape[1]==len(features) else vals[:,0,:]
    plt.figure(figsize=(10,6)); shap.summary_plot(vals,X,show=False); st.pyplot(plt.gcf()); plt.close()
    imp=pd.DataFrame({"Feature":features,"Mean_SHAP":np.abs(vals).mean(axis=0)}).sort_values("Mean_SHAP",ascending=False)
    st.dataframe(imp,use_container_width=True)
    st.plotly_chart(px.bar(imp,x="Mean_SHAP",y="Feature",orientation="h",color="Mean_SHAP"),use_container_width=True)
    idx=st.slider("Observation",0,len(X)-1,0)
    pred=model.predict(X.iloc[[idx]])[0]
    st.success(f"Predicted Class: {encoder.inverse_transform([pred])[0]}")
    try:
        plt.figure(figsize=(8,6)); shap.plots.waterfall(sv[idx],show=False); st.pyplot(plt.gcf()); plt.close()
    except Exception as e:
        st.warning(str(e))
