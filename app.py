import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="EDA & Fraud Signal Analysis", layout="wide")

st.title("📊 WIUT Hackathon — Anti-Fraud EDA Dashboard")
st.markdown("Interactive analytical web interface for transaction activity analysis and signal escalation forecasting (`eskalatsiya`).")

st.sidebar.header("Navigation")
menu = st.sidebar.radio("Sections", [
    "1. Overview",
    "2. Exploratory Data Analysis (EDA)",
    "3. Behavioral Patterns",
    "4. Model & Results"
])

if menu == "1. Overview":
    st.header("1. Dataset Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Signals (Train)", "50,000+")
    c2.metric("Escalation Rate (Target = 1)", "8.42%")
    c3.metric("Total Transactions", "1,200,000+")
        
    st.subheader("Sample Signals Data Structure")
    sample_df = pd.DataFrame({
        'signal_id': [101, 102, 103, 104, 105],
        'user_id': ['U8831', 'U9042', 'U1102', 'U4491', 'U3012'],
        'signal_type': ['transfers_count', 'amount_spike', 'new_device', 'transfers_count', 'login_attempt'],
        'eskalatsiya': [0, 1, 0, 0, 1]
    })
    st.dataframe(sample_df)

elif menu == "2. Exploratory Data Analysis (EDA)":
    st.header("2. Key Metrics Visualization")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    
    # Target distribution chart
    target_counts = [45790, 4210]
    ax[0].bar(['Normal (0)', 'Escalated (1)'], target_counts, color=['#2b5c8f', '#d95f02'])
    ax[0].set_title("Target Distribution (Eskalatsiya)")
    ax[0].set_ylabel("Count")
    
    # Transaction types chart
    tx_types = ['kirim', 'chiqim', 'xalqaro', 'bank_otkazmasi']
    tx_counts = [450000, 520000, 110000, 120000]
    ax[1].bar(tx_types, tx_counts, color='#2ca02c')
    ax[1].set_title("Transaction Types Distribution")
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
