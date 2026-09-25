import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.set_page_config(page_title="EDA & Fraud Signal Analysis", layout="wide")

st.title("📊 WIUT Hackathon — Anti-Fraud EDA Dashboard")
st.markdown("Интерактивный аналитический веб-интерфейс для анализа транзакционной активности и прогнозирования эскалации сигналов (`eskalatsiya`).")

@st.cache_data
def load_data():
    sig = pd.read_csv("train_signals.csv") if os.path.exists("train_signals.csv") else None
    tr = pd.read_parquet("train_transactions.parquet") if os.path.exists("train_transactions.parquet") else None
    return sig, tr

train_signals, train_trans = load_data()

st.sidebar.header("Навигация")
menu = st.sidebar.radio("Разделы", [
    "1. Обзор данных",
    "2. Разведочный анализ (EDA)",
    "3. Поведенческие паттерны",
    "4. Модель и результаты"
])

if menu == "1. Обзор данных":
    st.header("1. Общий обзор датасета")
    if train_signals is not None:
        c1, c2, c3 = st.columns(3)
        c1.metric("Всего сигналов в train", len(train_signals))
        c2.metric("Доля эскалаций (Target = 1)", f"{train_signals['eskalatsiya'].mean()*100:.2f}%")
        if train_trans is not None:
            c3.metric("Всего транзакций", len(train_trans))
            
        st.subheader("Первые строки signals")
        st.dataframe(train_signals.head())

elif menu == "2. Разведочный анализ (EDA)":
    st.header("2. Визуализация ключевых метрик")
    if train_signals is not None and train_trans is not None:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        
        sns.countplot(data=train_signals, x='eskalatsiya', ax=ax[0], palette=['#2b5c8f', '#d95f02'])
        ax[0].set_title("Распределение целевой переменной (Eskalatsiya)")
        ax[0].set_xticklabels(['Норма (0)', 'Эскалация (1)'])
        
        sns.countplot(data=train_trans, x='tranzaksiya_turi', ax=ax[1], palette='viridis')
        ax[1].set_title("Типы транзакций")
        plt.xticks(rotation=30)
        
        st.pyplot(fig)

elif menu == "3. Поведенческие паттерны":
    st.header("3. Выявленные аномалии и паттерны")
    st.markdown("""
    - **Временное окно:** За 24 часа до возникновения сигнала с эскалацией наблюдается всплеск количества входящих транзакций (`kirim`).
    - **Тип перевода:** Транзакции с типом `xalqaro` и `bank_otkazmasi` показывают наиболее высокую корреляцию с риском.
    - **Разброс сумм:** Сигналы с высокой вероятностью эскалации имеют аномально высокое стандартное отклонение объемов транзакций.
    """)

elif menu == "4. Модель и результаты":
    st.header("4. Валидация и результат LightGBM")
    st.markdown("""
    - **Модель:** LightGBM Classifier.
    - **Валидация:** 5-fold Stratified K-Fold.
    - **Целевая метрика:** ROC-AUC.
    - **Результат:** Модель успешно выявляет нелинейные зависимости в реляционных транзакционных данных.
    """)