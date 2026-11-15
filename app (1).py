# ============================================================
# TITANIC DATA ANALYTICS DASHBOARD USING STREAMLIT
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Titanic Analytics Dashboard",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# 2. CUSTOM CSS
# ============================================================

st.markdown("""
<style>
.main {
    background-color: #f5f7fb;
}
[data-testid="stMetric"] {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #e5e7eb;
}
h1 {
    color: #16324f;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# 3. LOAD DATASET
# ============================================================

@st.cache_data
def load_data():
    url = (
        "https://raw.githubusercontent.com/"
        "datasciencedojo/datasets/master/titanic.csv"
    )
    return pd.read_csv(url)


try:
    df = load_data()
except Exception as e:
    st.error(f"Dataset loading failed: {e}")
    st.stop()

# ============================================================
# 4. DATA PREPROCESSING
# ============================================================

df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df["Fare"] = df["Fare"].fillna(df["Fare"].median())
df["Cabin"] = df["Cabin"].fillna("Unknown")

df["Survival_Status"] = df["Survived"].map({
    0: "Not Survived",
    1: "Survived"
})
df["Gender"] = df["Sex"].str.title()
df["Pclass"] = df["Pclass"].astype(str)

# ============================================================
# 5. DASHBOARD HEADER
# ============================================================

st.title("🚢 Titanic Passenger Analytics Dashboard")
st.markdown(
    "Explore passenger demographics, survival patterns, "
    "ticket classes, fares and passenger characteristics."
)
st.divider()

# ============================================================
# 6. SIDEBAR FILTERS
# ============================================================

st.sidebar.title("Dashboard Filters")

gender_options = sorted(df["Gender"].dropna().unique().tolist())
class_options = sorted(df["Pclass"].dropna().unique().tolist())
embarked_options = sorted(df["Embarked"].dropna().unique().tolist())
survival_options = df["Survival_Status"].dropna().unique().tolist()

gender_filter = st.sidebar.multiselect(
    "Select Gender",
    options=gender_options,
    default=gender_options
)

class_filter = st.sidebar.multiselect(
    "Select Passenger Class",
    options=class_options,
    default=class_options
)

embarked_filter = st.sidebar.multiselect(
    "Select Embarkation Port",
    options=embarked_options,
    default=embarked_options
)

survival_filter = st.sidebar.multiselect(
    "Survival Status",
    options=survival_options,
    default=survival_options
)

age_min = int(df["Age"].min())
age_max = int(df["Age"].max())

age_range = st.sidebar.slider(
    "Select Age Range",
    min_value=age_min,
    max_value=age_max,
    value=(age_min, age_max)
)

# ============================================================
# 7. APPLY FILTERS
# ============================================================

filtered_df = df[
    df["Gender"].isin(gender_filter)
    & df["Pclass"].isin(class_filter)
    & df["Embarked"].isin(embarked_filter)
    & df["Survival_Status"].isin(survival_filter)
    & df["Age"].between(age_range[0], age_range[1])
].copy()

if filtered_df.empty:
    st.warning("No passengers match the selected filters. Please adjust them.")
    st.stop()

# ============================================================
# 8. KPI METRICS
# ============================================================

st.subheader("📊 Key Performance Indicators")

total_passengers = len(filtered_df)
survivors = int(filtered_df["Survived"].sum())
survival_rate = survivors / total_passengers * 100
average_age = filtered_df["Age"].mean()
average_fare = filtered_df["Fare"].mean()

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Total Passengers", f"{total_passengers:,}")
col2.metric("Survivors", f"{survivors:,}")
col3.metric("Survival Rate", f"{survival_rate:.2f}%")
col4.metric("Average Age", f"{average_age:.1f}")
col5.metric("Average Fare", f"${average_fare:.2f}")

st.divider()

# ============================================================
# 9. GENDER ANALYSIS
# ============================================================

st.subheader("👥 Gender-wise Analysis")
col1, col2 = st.columns(2)

with col1:
    gender_count = (
        filtered_df["Gender"].value_counts()
        .rename_axis("Gender")
        .reset_index(name="Passengers")
    )

    fig = px.pie(
        gender_count,
        names="Gender",
        values="Passengers",
        hole=0.4,
        title="Passenger Distribution by Gender"
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    gender_survival = (
        filtered_df.groupby("Gender", as_index=False)["Survived"].mean()
    )
    gender_survival["Survival Rate"] = gender_survival["Survived"] * 100

    fig = px.bar(
        gender_survival,
        x="Gender",
        y="Survival Rate",
        color="Gender",
        text_auto=".2f",
        title="Survival Rate by Gender"
    )
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 10. PASSENGER CLASS ANALYSIS
# ============================================================

st.subheader("🎫 Passenger Class Analysis")
col1, col2 = st.columns(2)

with col1:
    class_survival = (
        filtered_df.groupby("Pclass", as_index=False)["Survived"].mean()
    )
    class_survival["Survival Rate"] = class_survival["Survived"] * 100

    fig = px.bar(
        class_survival,
        x="Pclass",
        y="Survival Rate",
        color="Pclass",
        text_auto=".2f",
        title="Survival Rate by Passenger Class"
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.histogram(
        filtered_df,
        x="Pclass",
        color="Survival_Status",
        barmode="group",
        title="Passenger Count by Class and Survival"
    )
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 11. AGE ANALYSIS
# ============================================================

st.subheader("🎂 Age Distribution Analysis")
col1, col2 = st.columns(2)

with col1:
    fig = px.histogram(
        filtered_df,
        x="Age",
        nbins=30,
        color="Survival_Status",
        title="Age Distribution by Survival Status"
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.box(
        filtered_df,
        x="Gender",
        y="Age",
        color="Gender",
        title="Age Distribution by Gender"
    )
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 12. FARE ANALYSIS
# ============================================================

st.subheader("💰 Fare Analysis")

fig = px.histogram(
    filtered_df,
    x="Fare",
    nbins=40,
    color="Survival_Status",
    title="Fare Distribution by Survival Status"
)
st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 13. EMBARKATION ANALYSIS
# ============================================================

st.subheader("⚓ Embarkation Analysis")
col1, col2 = st.columns(2)

with col1:
    embark_count = (
        filtered_df["Embarked"].value_counts()
        .rename_axis("Port")
        .reset_index(name="Passengers")
    )

    fig = px.pie(
        embark_count,
        names="Port",
        values="Passengers",
        title="Passengers by Embarkation Port"
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    embark_survival = (
        filtered_df.groupby("Embarked", as_index=False)["Survived"].mean()
    )
    embark_survival["Survival Rate"] = embark_survival["Survived"] * 100

    fig = px.bar(
        embark_survival,
        x="Embarked",
        y="Survival Rate",
        color="Embarked",
        text_auto=".2f",
        title="Survival Rate by Embarkation Port"
    )
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 14. CORRELATION HEATMAP
# ============================================================

st.subheader("🔥 Correlation Analysis")

numeric_df = filtered_df.select_dtypes(include=np.number)
corr = numeric_df.corr()

fig, ax = plt.subplots(figsize=(12, 7))
sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    ax=ax
)
ax.set_title("Correlation Heatmap")
st.pyplot(fig)
plt.close(fig)

# ============================================================
# 15. DATA TABLE
# ============================================================

st.subheader("📋 Passenger Data Explorer")
st.write(f"Showing {len(filtered_df)} records")

st.dataframe(
    filtered_df,
    use_container_width=True,
    hide_index=True
)

# ============================================================
# 16. MISSING VALUE ANALYSIS
# ============================================================

st.subheader("🔍 Missing Value Analysis")

missing_df = df.isnull().sum().reset_index()
missing_df.columns = ["Column", "Missing Values"]
missing_df["Missing Percentage"] = (
    missing_df["Missing Values"] / len(df) * 100
)

st.dataframe(
    missing_df,
    use_container_width=True,
    hide_index=True
)

# ============================================================
# 17. DOWNLOAD FILTERED DATASET
# ============================================================

st.subheader("📥 Download Dataset")

csv = filtered_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Filtered Dataset",
    data=csv,
    file_name="titanic_filtered_data.csv",
    mime="text/csv"
)

# ============================================================
# 18. FOOTER
# ============================================================

st.divider()
st.markdown(
    """
    <div style="text-align: center;">
        <b>Titanic Analytics Dashboard</b><br>
        Built using Python | Pandas | Streamlit | Plotly | Seaborn
    </div>
    """,
    unsafe_allow_html=True
)
