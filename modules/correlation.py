from textwrap import dedent
content=dedent("""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

def correlation_dashboard(df):
    st.header("🔥 Climate Variable Correlation Analysis")
    climate_variables=["Rainfall","Tmax","Tmin","Humidity","Pressure","Wind","Solar","ENSO","IOD","NDVI","SPI"]
    selected=st.multiselect("Select Variables",climate_variables,default=climate_variables)
    if len(selected)<2:
        st.warning("Please select at least two variables.")
        return
    corr=df[selected].corr(method="pearson")
    fig=px.imshow(corr,text_auto=".2f",color_continuous_scale="RdBu_r",zmin=-1,zmax=1,aspect="auto")
    fig.update_layout(height=700,title="Pearson Correlation Matrix")
    st.plotly_chart(fig,use_container_width=True)
    st.subheader("Correlation Matrix")
    st.dataframe(corr,use_container_width=True)
    corr2=corr.abs().copy()
    np.fill_diagonal(corr2.values,np.nan)
    top=corr2.stack().reset_index()
    top.columns=["Variable 1","Variable 2","Correlation"]
    top["pair"]=top.apply(lambda x: tuple(sorted([x["Variable 1"],x["Variable 2"]])),axis=1)
    top=top.drop_duplicates("pair").drop(columns="pair").sort_values("Correlation",ascending=False)
    st.subheader("Top Correlation")
    st.dataframe(top.head(10),use_container_width=True)
    values=corr.values.flatten()
    values=values[~np.isnan(values)]
    values=values[np.abs(values)<0.999]
    hist=px.histogram(x=values,nbins=30,color_discrete_sequence=["royalblue"])
    st.plotly_chart(hist,use_container_width=True)
    strong=[]
    for i in range(len(selected)):
        for j in range(i+1,len(selected)):
            r=corr.iloc[i,j]
            if abs(r)>=0.70:
                strong.append([selected[i],selected[j],round(r,3)])
    st.subheader("Strong Correlation")
    if strong:
        st.dataframe(pd.DataFrame(strong,columns=["Variable A","Variable B","Correlation"]),use_container_width=True)
    else:
        st.info("No strong correlation detected.")
    st.download_button("📥 Download Correlation Matrix",corr.to_csv(),"correlation_matrix.csv","text/csv")
""")
path="/mnt/data/correlation.py"
with open(path,"w",encoding="utf-8") as f:
    f.write(content)
print(path)
