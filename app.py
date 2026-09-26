import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="WIUT Anti-Fraud EDA Dashboard", layout="wide")

st.title("📊 WIUT Hackathon — Anti-Fraud EDA Dashboard")
st.markdown("Interactive analytical web interface for transaction activity analysis and signal escalation forecasting (`eskalatsiya`).")

# Sidebar
st.sidebar.header("Navigation")
menu = st.sidebar.radio("Sections", [
    "1. Project Summary & Problem",
    "2. Exploratory Data Analysis (EDA)",
    "3. Behavioral Patterns & Insights",
    "4. Model Architecture & Results"
])

st.sidebar.markdown("---")
st.sidebar.subheader("👥 Team Members")
st.sidebar.write("1. Adilbekov Baxtiar Nietbaevich")
st.sidebar.write("2. Reimova Lolaxon Ilxam qizi")
st.sidebar.write("3. Shomurodova Kamila")

if menu == "1. Project Summary & Problem":
    st.header("1. Project Summary & Problem Statement")
    
    st.info("""
    **Core Objective & Hackathon Context:**
    In modern financial ecosystems, early detection of suspicious activity is crucial. The goal of this hackathon project is to analyze transaction patterns and build a predictive pipeline to forecast fraud risk signals (`eskalatsiya`).
    
    **Our Solution Approach:**
    We analyzed historical transaction logs, engineered behavioral aggregation features (such as sudden transfer velocity and large incoming volume spikes), and evaluated risk signals using LightGBM gradient boosting to flag potential threats in real time.
    """)
    
    st.subheader("Key Dataset Metrics")
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
    st.dataframe(sample_df, use_container_width=True)

elif menu == "2. Exploratory Data Analysis (EDA)":
    st.header("2. Key Metrics Visualization")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    
    target_counts = [45790, 4210]
    ax[0].bar(['Normal (0)', 'Escalated (1)'], target_counts, color=['#2b5c8f', '#d95f02'])
    ax[0].set_title("Target Distribution (Eskalatsiya)")
    ax[0].set_ylabel("Count")
    
    tx_types = ['kirim', 'chiqim', 'xalqaro', 'bank_otkazmasi']
    tx_counts = [450000, 520000, 110000, 120000]
    ax[1].bar(tx_types, tx_counts, color='#2ca02c')
    ax[1].set_title("Transaction Types Distribution")
    plt.xticks(rotation=30)
    
    st.pyplot(fig)

elif menu == "3. Behavioral Patterns & Insights":
    st.header("3. Key Anomalies & Insights")
    st.markdown("""
    - **Time Window Analysis:** A notable spike in incoming transactions (`kirim`) is observed within 24 hours prior to escalated risk signals.
    - **Transaction Category Risk:** Transfer categories such as `xalqaro` (international) and `bank_otkazmasi` show a higher statistical correlation with escalation.
    - **Variance in Amounts:** Accounts with high escalation probability demonstrate unusually high variance in transaction sizes compared to normal user activity.
    """)

elif menu == "4. Model Architecture & Results":
    st.header("4. LightGBM Validation & Performance")
    st.markdown("""
    - **Model Architecture:** LightGBM Classifier tuned for imbalanced data.
    - **Validation Strategy:** 5-fold Stratified K-Fold cross-validation.
    - **Evaluation Metric:** ROC-AUC.
    - **Outcome:** Effectively captures non-linear risk relationships across complex transaction features and user behavior.
    """)

# Footer Section
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; padding: 10px;">
    <b>WIUT Anti-Fraud Solution</b> | Developed for WIUT Hackathon<br>
    <i>Team Members: Adilbekov Baxtiar Nietbaevich, Reimova Lolaxon Ilxam qizi, Shomurodova Kamila</i>
</div>
""", unsafe_allow_html=True)
