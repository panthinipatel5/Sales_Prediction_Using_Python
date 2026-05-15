import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# ---------------- PAGE SETTINGS ----------------

st.set_page_config(
    page_title="Sales Prediction Dashboard",
    page_icon="📈",
    layout="wide"
)

# ---------------- TITLE ----------------

st.title("📈 Sales Prediction Dashboard")
st.markdown("Predict Product Sales using Machine Learning")

# ---------------- LOAD DATA ----------------

df = pd.read_csv("Advertising.csv")

# ---------------- LOAD MODEL ----------------

model = joblib.load("sales_model.pkl")

# ---------------- SIDEBAR ----------------

st.sidebar.header("Enter Advertisement Budget")

tv = st.sidebar.slider("TV Advertisement Budget", 0.0, 300.0, 150.0)

radio = st.sidebar.slider("Radio Advertisement Budget", 0.0, 50.0, 25.0)

newspaper = st.sidebar.slider("Newspaper Advertisement Budget", 0.0, 120.0, 30.0)

# ---------------- PREDICTION ----------------

input_data = np.array([[tv, radio, newspaper]])

prediction = model.predict(input_data)

# ---------------- DISPLAY PREDICTION ----------------

st.subheader("📊 Predicted Sales")

st.success(f"Estimated Sales: {prediction[0]:.2f}")

# ---------------- DATA PREVIEW ----------------

st.subheader("📋 Dataset Preview")

st.dataframe(df.head())

# ---------------- DATA INFO ----------------

st.subheader("📌 Dataset Shape")

st.write("Rows:", df.shape[0])
st.write("Columns:", df.shape[1])

# ---------------- STATISTICS ----------------

st.subheader("📈 Statistical Summary")

st.dataframe(df.describe())

# ---------------- CORRELATION HEATMAP ----------------

st.subheader("🔥 Correlation Heatmap")

fig_heatmap = px.imshow(
    df.corr(),
    text_auto=True,
    aspect="auto"
)

st.plotly_chart(fig_heatmap, use_container_width=True)

# ---------------- TV VS SALES ----------------

st.subheader("📺 TV Advertisement vs Sales")

fig_tv = px.scatter(
    df,
    x="TV",
    y="Sales",
    trendline="ols"
)

st.plotly_chart(fig_tv, use_container_width=True)

# ---------------- RADIO VS SALES ----------------

st.subheader("📻 Radio Advertisement vs Sales")

fig_radio = px.scatter(
    df,
    x="Radio",
    y="Sales",
    trendline="ols"
)

st.plotly_chart(fig_radio, use_container_width=True)

# ---------------- NEWSPAPER VS SALES ----------------

st.subheader("📰 Newspaper Advertisement vs Sales")

fig_news = px.scatter(
    df,
    x="Newspaper",
    y="Sales",
    trendline="ols"
)

st.plotly_chart(fig_news, use_container_width=True)

# ---------------- FOOTER ----------------

st.markdown("---")

st.markdown("✅ Built with Python, Streamlit & Machine Learning")