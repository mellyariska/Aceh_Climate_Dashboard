import os
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,classification_report
import joblib

def random_forest_dashboard(df):
    st.header("🌳 Random Forest Machine Learning")
    feats=["Rainfall","Tmax","Tmin","Humidity","Pressure","Wind","Solar","ENSO","IOD","NDVI","SPI"]
    X=df[feats].apply(pd.to_numeric,errors="coerce")
    y=df["Disaster"]
    data=pd.concat([X,y],axis=1).dropna()
    X=data[feats]; y=data["Disaster"]
    n=st.slider("Number of Trees",50,500,200,10)
    d=st.slider("Maximum Depth",2,30,10)
    if y.value_counts().min()<2:
        Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.3,random_state=42)
    else:
        Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.3,random_state=42,stratify=y)
    m=RandomForestClassifier(n_estimators=n,max_depth=d,random_state=42,n_jobs=-1)
    m.fit(Xtr,ytr); pred=m.predict(Xte)
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Accuracy",f"{accuracy_score(yte,pred):.3f}")
    c2.metric("Precision",f"{precision_score(yte,pred,average='weighted',zero_division=0):.3f}")
    c3.metric("Recall",f"{recall_score(yte,pred,average='weighted',zero_division=0):.3f}")
    c4.metric("F1",f"{f1_score(yte,pred,average='weighted',zero_division=0):.3f}")
    imp=pd.DataFrame({"Feature":feats,"Importance":m.feature_importances_}).sort_values("Importance",ascending=False)
    st.plotly_chart(px.bar(imp,x="Importance",y="Feature",orientation="h"),use_container_width=True)
    st.plotly_chart(px.imshow(confusion_matrix(yte,pred,labels=m.classes_),text_auto=True,x=m.classes_,y=m.classes_),use_container_width=True)
    st.dataframe(pd.DataFrame(classification_report(yte,pred,output_dict=True,zero_division=0)).transpose(),use_container_width=True)
    os.makedirs("models",exist_ok=True)
    p="models/random_forest.pkl"; joblib.dump(m,p)
    with open(p,"rb") as f: st.download_button("📥 Download Random Forest Model",f,file_name="random_forest.pkl")
    st.download_button("📥 Download Feature Importance",imp.to_csv(index=False),"feature_importance.csv","text/csv")
