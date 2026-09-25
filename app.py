import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.set_page_config(page_title="EDA & Fraud Signal Analysis", layout="wide")

st.title("📊 WIUT Hackathon — Anti-Fraud EDA Dashboard")
st.markdown("Interactive analytical web interface for transaction activity analysis and signal escalation forecasting (`eskalatsiya`).")

@st.cache_data
def load_data():
    sig = pd.read_csv("train_signals.csv") if os.path.exists("train_signals.csv") else None
    tr = pd.read_parquet("train_transactions.parquet") if os.path.exists("train_transactions.parquet") else None
    return sig, tr

train_signals, train_trans = load_data()

st.sidebar.header("Navigation")
menu = st.sidebar.radio("Sections", [
    "1. Overview",
    "2. Exploratory Data Analysis (EDA)",
    "3. Behavioral Patterns",
    "4. Model & Results"
])

if menu == "1. Overview":
    st.header("1. Dataset Overview")
    if train_signals is not None:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Signals (Train)", len(train_signals))
        c2.metric("Escalation Rate (Target = 1)", f"{train_signals['eskalatsiya'].mean()*100:.2f}%")
        if train_trans is not None:
            c3.metric("Total Transactions", len(train_trans))
            
        st.subheader("Sample Signals Data")
        st.dataframe(train_signals.head())

elif menu == "2. Exploratory Data Analysis (EDA)":
    st.header("2. Key Metrics Visualization")
    if train_signals is not None and train_trans is not None:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        
        sns.countplot(data=train_signals, x='eskalatsiya', ax=ax[0], palette=['#2b5c8f', '#d95f02'])
        ax[0].set_title("Target Distribution (Eskalatsiya)")
        ax[0].set_xticklabels(['Normal (0)', 'Escalated (1)'])
        
        sns.countplot(data=train_trans, x='tranzaksiya_turi', ax=ax[1], palette='viridis')
        ax[1].set_title("Transaction Types")
        plt.xticks(rotation=30)
        
        st.pyplot(fig)

elif menu == "3. Behavioral Patterns":
    st.header("3. Key Anomalies & Insights")
    st.markdown("""
    - **Time Window:** High spike in incoming transactions (`kirim`) 24 hours prior to escalated signals.
    - **Transfer Types:** `xalqaro` and `bank_otkazmasi` transactions show the highest correlation with risk.
    - **Amount Variance:** Signals with high escalation probability exhibit abnormally high standard deviation in transaction volumes.
    """)

elif menu == "4. Model & Results":
    st.header("4. LightGBM Validation & Performance")
    st.markdown("""
    - **Model Architecture:** LightGBM Classifier.
    - **Validation Scheme:** 5-fold Stratified K-Fold.
    - **Evaluation Metric:** ROC-AUC.
    - **Outcome:** Successfully captures non-linear relational patterns within transaction features.
    """)
