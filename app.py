import streamlit as st
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# 設定網頁標題
st.set_page_config(page_title="酒類預測 ML 專題", layout="wide")

# 1. 載入資料集
@st.cache_data
def load_data():
    wine = datasets.load_wine()
    df = pd.DataFrame(wine.data, columns=wine.feature_names)
    df['target'] = wine.target
    return wine, df

wine_data, df = load_data()

# 2. 左側 Sidebar
st.sidebar.header("模型設定")
model_name = st.sidebar.selectbox(
    "選擇模型",
    ("KNN", "羅吉斯迴歸", "XGBoost", "隨機森林")
)

st.sidebar.markdown("---")
st.sidebar.header("資料集資訊: 酒類 (Wine)")
st.sidebar.info(f"""
**描述**: 這是著名的 Scikit-learn 酒類資料集。
- **特徵數量**: {len(wine_data.feature_names)}
- **樣本數量**: {len(df)}
- **目標類別**: {len(wine_data.target_names)} (類別: {', '.join(wine_data.target_names)})
""")

# 3. 右側 Main 區
st.title("🍷 酒類分類預測系統")

st.subheader("📊 資料集前 5 筆內容")
st.dataframe(df.head())

st.subheader("📈 特徵統計值資訊")
st.write(df.describe())

# 4. 預測邏輯
st.markdown("---")
st.subheader(f"🚀 模型預測: {model_name}")

if st.button("進行預測並顯示結果"):
    # 分割資料
    X = wine_data.data
    y = wine_data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 選擇模型
    if model_name == "KNN":
        model = KNeighborsClassifier(n_neighbors=5)
    elif model_name == "羅吉斯迴歸":
        model = LogisticRegression(max_iter=10000)
    elif model_name == "XGBoost":
        model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    elif model_name == "隨機森林":
        model = RandomForestClassifier(n_estimators=100, random_state=42)

    # 訓練模型
    model.fit(X_train, y_train)

    # 進行預測
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # 顯示結果
    col1, col2 = st.columns(2)
    with col1:
        st.metric("模型準確度 (Accuracy)", f"{acc:.2%}")
    with col2:
        st.success(f"已使用 {model_name} 模型完成預測！")
    
    st.write("測試集預測結果 (前 10 筆):")
    results_df = pd.DataFrame({
        '實際值': [wine_data.target_names[i] for i in y_test[:10]],
        '預測值': [wine_data.target_names[i] for i in y_pred[:10]]
    })
    st.table(results_df)

st.markdown("---")
st.caption("Developed by Antigravity AI")
