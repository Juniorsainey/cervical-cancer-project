import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# ---- CHART STYLING (matches dashboard theme) ----
plt.rcParams.update({
    'figure.facecolor': '#1A2129',
    'axes.facecolor': '#1A2129',
    'axes.edgecolor': '#EAEAEA',
    'axes.labelcolor': '#EAEAEA',
    'text.color': '#EAEAEA',
    'xtick.color': '#EAEAEA',
    'ytick.color': '#EAEAEA',
    'grid.color': '#2A3139',
    'font.size': 11,
})

CORAL = '#FF6B5B'
NAVY_LIGHT = '#3D8B8A'

# ---- PAGE SETUP ----
st.set_page_config(page_title="Cervical Cancer Analysis", layout="wide", page_icon="🏥")

# ---- LOAD DATA ----
@st.cache_data
def load_data():
    df = pd.read_csv('data/risk_factors_cervical_cancer.csv')
    df = df.replace('?', np.nan)
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.fillna(df.median())
    return df

df = load_data()

# ---- SIDEBAR ----
st.sidebar.title("🏥 Navigation")
page = st.sidebar.radio("Go to:", ["Overview", "Explore Data", "Model & Results"])

st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Filter Patients")
age_range = st.sidebar.slider("Age Range", int(df['Age'].min()), int(df['Age'].max()), (int(df['Age'].min()), int(df['Age'].max())))
smoker_filter = st.sidebar.selectbox("Smoking Status", ["All", "Smokers", "Non-Smokers"])

# Apply filters
filtered_df = df[(df['Age'] >= age_range[0]) & (df['Age'] <= age_range[1])]
if smoker_filter == "Smokers":
    filtered_df = filtered_df[filtered_df['Smokes'] == 1]
elif smoker_filter == "Non-Smokers":
    filtered_df = filtered_df[filtered_df['Smokes'] == 0]

st.sidebar.markdown(f"**Showing {len(filtered_df)} of {len(df)} patients**")
st.sidebar.markdown("---")
st.sidebar.caption("University of The Gambia | Data Engineering Project")

# ---- MAIN CONTENT ----

if page == "Overview":
    st.title("🏥 Cervical Cancer Risk Factors Analysis")
    st.markdown("**University of The Gambia | Data Engineering Project**")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Patients", len(df))
    col2.metric("Filtered Patients", len(filtered_df))
    col3.metric("Variables", df.shape[1])
    col4.metric("Cancer Cases", int(df['Biopsy'].sum()))

    st.markdown("### 📋 Sample Data")
    st.dataframe(filtered_df.head(10))

    st.markdown("### 📊 Quick Stats")
    st.dataframe(filtered_df.describe())

elif page == "Explore Data":
    st.title("📊 Exploratory Data Analysis")
    st.markdown(f"*Currently showing {len(filtered_df)} patients based on your filters*")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Age Distribution")
        fig1, ax1 = plt.subplots()
        sns.histplot(filtered_df['Age'], bins=20, color=CORAL, ax=ax1)
        st.pyplot(fig1)

    with col2:
        st.subheader("Biopsy Results")
        fig2, ax2 = plt.subplots()
        sns.countplot(x='Biopsy', data=filtered_df, palette=[NAVY_LIGHT, CORAL], ax=ax2)
        ax2.set_xlabel('Biopsy (0=No Cancer, 1=Cancer)')
        st.pyplot(fig2)

    st.subheader("🔥 Correlation Heatmap")
    fig3, ax3 = plt.subplots(figsize=(15,10))
    sns.heatmap(df.corr(), cmap='rocket_r', ax=ax3)
    st.pyplot(fig3)

elif page == "Model & Results":
    st.title("🤖 Model Training & Evaluation")
    st.markdown("---")

    X = df.drop('Biopsy', axis=1)
    y = df['Biopsy']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    acc = (y_pred == y_test).mean()
    st.metric("Model Accuracy", f"{acc*100:.1f}%")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())

    with col2:
        st.subheader("Confusion Matrix")
        fig4, ax4 = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='rocket_r',
                    xticklabels=['No Cancer','Cancer'], yticklabels=['No Cancer','Cancer'], ax=ax4)
        st.pyplot(fig4)