import streamlit as st
import pandas as pd
import plotly.express as px

st.title("WORKING DASHBOARD")
st.success("CALISIYORUM!")

data = {"Station": ["A", "B", "C"], "Score": [0.8, 0.7, 0.9]}
df = pd.DataFrame(data)
st.dataframe(df)

fig = px.bar(df, x="Station", y="Score")
st.plotly_chart(fig)