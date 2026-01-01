import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error as MSE, r2_score

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")


st.set_page_config(page_title="Lin Regression App", layout='wide')
st.title("Lin Regression")

section = st.sidebar.radio(
    "Раздел",
    ["EDA", "Предсказание", "Веса модели"]
)

def fix_datatypes(df: pd.DataFrame, cols: list):
    for col in cols:
        df[col] = (
            df[col]
            .str.extract(r'([\d\.]+)')
            .astype(float)
        )
    return df
    
def fill_nans(df: pd.DataFrame, cols: list):
    for col in cols:
        median = df[col].median()
        df[col] = df[col].fillna(median)

    return df

def fix_df(df: pd.DataFrame):
    df.drop('torque', axis=1, inplace=True)
    df = df.loc[~df.duplicated(keep='first')]

    df = fix_datatypes(df, ['max_power', 'engine', 'mileage'])
    df = fill_nans(df, ['max_power', 'engine', 'mileage', 'seats'])

    df['engine'] = df['engine'].astype(int)
    df['seats'] = df['seats'].astype(int)

    return df

if section == "EDA":

    st.header("EDA")

    uploaded = st.file_uploader("Загрузите CSV для EDA", type="csv")
    if uploaded:
        df = pd.read_csv(uploaded)

        st.subheader("Первые строки")
        st.dataframe(df.head())

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Пропуски")
            missing = df.isna().sum()
            missing = missing[missing > 0].sort_values(ascending=False)
            st.dataframe(missing)
        with col2:
            st.subheader("Дубликаты")
            st.metric("Дубликаты", df.duplicated().sum())

        df = fix_df(df)
        st.text("Далее пропуски пофиксили, дубликаты тоже, удалили torque")

        st.subheader("Pairplot")
        num_cols = df.select_dtypes(include=np.number).columns.tolist()

        selected_cols = st.multiselect(
            "Выберите признаки",
            num_cols,
            default=num_cols[:4]
        )

        if selected_cols:
            plot_df = df[selected_cols].dropna()

            if len(plot_df) > 1000:
                plot_df = plot_df.sample(1000, random_state=42)

            fig = sns.pairplot(plot_df, height=1.8)
            st.pyplot(fig, use_container_width=False)

        st.subheader("Корреляции")
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(df.select_dtypes(include='number').corr(), cmap="Blues", annot=True)
        st.pyplot(fig, use_container_width=False)


elif section == "Предсказание":
    st.header("Предсказание")

    mode = st.radio("Режим", ["CSV", "Ручной ввод"])

    if mode == "CSV":
        uploaded = st.file_uploader("CSV", type="csv")
        if uploaded:
            df = pd.read_csv(uploaded)
            df = fix_df(df)
            
            if 'selling_price' not in df.columns:
                st.error("В CSV должна быть колонка 'selling_price'")
            else:
                y_true = df['selling_price']
                df.drop('selling_price', inplace=True, axis=1)

                df_num = df.select_dtypes(include='number')
                X_scaled = pd.DataFrame(scaler.transform(df_num), columns=df_num.columns, index=df_num.index)
                preds = model.predict(X_scaled)

                st.subheader("Метрики на CSV")
                st.write(f"MSE = {MSE(y_true, preds):,.2f}")
                st.write(f"R2 = {r2_score(y_true, preds):.4f}")
    else:
        num_features = {}
        num_features["year"] = st.number_input("Год выпуска", min_value=1900, max_value=2030, value=2015)
        num_features["km_driven"] = st.number_input("Пробег (км)", min_value=0, max_value=1_000_000, value=50000)
        num_features["mileage"] = st.number_input("Милиаж (км/л)", min_value=0.0, max_value=100.0, value=18.0)
        num_features["engine"] = st.number_input("Объём двигателя (CC)", min_value=500, max_value=10000, value=1200)
        num_features["max_power"] = st.number_input("Мощность двигателя (bhp)", min_value=10.0, max_value=1000.0, value=75.0)
        num_features["seats"] = st.number_input("Количество мест", min_value=1, max_value=20, value=5)

        if st.button("Сделать предсказание"):
            input_df = pd.DataFrame([num_features])
            X_scaled = pd.DataFrame(scaler.transform(input_df), columns=input_df.columns, index=input_df.index)
            pred = model.predict(X_scaled)
            st.subheader("Предсказанная цена автомобиля")
            st.write(f"{pred[0]:,.2f}")


elif section == "Веса модели":
    features = ['year', 'km_driven', 'mileage', 'engine', 'max_power', 'seats']
    coefs = model.coef_

    fig, ax = plt.subplots(figsize=(8,4))
    bars = ax.bar(features, coefs, color='skyblue')
    ax.set_xlabel('Признаки')
    ax.set_ylabel('Веса')
    ax.set_title('Важность')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    for bar, coef in zip(bars, coefs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height, f'{coef:.2f}', ha='center', va='bottom')

    st.pyplot(fig)