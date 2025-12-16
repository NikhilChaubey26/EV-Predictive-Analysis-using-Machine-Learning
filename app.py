# app.py
# ========================================
# EV Predictive Analysis – Streamlit Dashboard
# ========================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_squared_error, r2_score, roc_curve, auc,
    precision_recall_fscore_support   # ✅ added for comparison metrics
)

# ------------------------------------------------
# 0. CONFIG
# ------------------------------------------------
st.set_page_config(page_title="EV Predictive Analysis", layout="wide")
DATA_PATH = r"C:\Users\nikhi\Downloads\Electric_Vehicle_Population_Data.csv"  # change if needed


# ------------------------------------------------
# 1. DATA LOADING + CLEANING (CACHED)
# ------------------------------------------------
@st.cache_data
def load_and_clean_data(path):
    df = pd.read_csv(path)

    # clean column names
    df.columns = (
        df.columns.str.strip()
                  .str.lower()
                  .str.replace(r'[^a-z0-9_]+', '_', regex=True)
                  .str.replace(r'_+', '_', regex=True)
                  .str.strip('_')
    )

    # parse location to lon/lat if present
    if "vehicle_location" in df.columns:
        coords = df["vehicle_location"].str.extract(r"POINT \(([-0-9\.]+) ([-0-9\.]+)\)")
        df["longitude"] = coords[0].astype(float)
        df["latitude"] = coords[1].astype(float)

    # treat 0 as missing for some numeric cols
    for col in ["electric_range", "base_msrp"]:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)

    df = df.drop_duplicates()

    # impute numeric
    for col in df.select_dtypes(include="number").columns:
        df[col] = df[col].fillna(df[col].median())

    # impute categorical
    for col in df.select_dtypes(include="object").columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    # drop ID-like columns
    df = df.drop(columns=["vin_1_10", "dol_vehicle_id", "vehicle_location"], errors="ignore")

    # final drop_duplicates
    df = df.drop_duplicates()

    # sample for speed
    if len(df) > 50000:
        df = df.sample(50000, random_state=42)

    return df


# ------------------------------------------------
# 2. MODEL TRAINING (CACHED)
# ------------------------------------------------
@st.cache_resource
def train_models(df: pd.DataFrame):
    results = {}

    # ---------- CLASSIFICATION ----------
    target = "clean_alternative_fuel_vehicle_cafv_eligibility"
    unknown = "Eligibility unknown as battery range has not been researched"

    df_clf = df[df[target] != unknown].copy()
    X_clf = df_clf.drop(columns=[target])
    y_clf = df_clf[target]

    X_clf = pd.get_dummies(X_clf, drop_first=True)

    Xc_train, Xc_test, yc_train, yc_test = train_test_split(
        X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
    )

    # logistic regression
    log_reg = LogisticRegression(max_iter=2000, class_weight="balanced")
    log_reg.fit(Xc_train, yc_train)
    yc_pred_log = log_reg.predict(Xc_test)
    yc_proba_log = log_reg.predict_proba(Xc_test)[:, 1] if len(log_reg.classes_) == 2 else None

    # random forest classifier
    rf_clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        class_weight="balanced",
        random_state=42
    )
    rf_clf.fit(Xc_train, yc_train)
    yc_pred_rf = rf_clf.predict(Xc_test)
    yc_proba_rf = rf_clf.predict_proba(Xc_test)[:, 1] if len(rf_clf.classes_) == 2 else None

    results["classification"] = {
        "X_test": Xc_test,
        "y_test": yc_test,
        "log_reg": log_reg,
        "rf_clf": rf_clf,
        "pred_log": yc_pred_log,
        "pred_rf": yc_pred_rf,
        "proba_log": yc_proba_log,
        "proba_rf": yc_proba_rf,
    }

    # ---------- REGRESSION ----------
    reg_target = "electric_range"
    reg_features = ["model_year", "base_msrp", "longitude", "latitude"]
    reg_features = [f for f in reg_features if f in df.columns]

    df_reg = df.dropna(subset=[reg_target]).copy()
    X_reg = df_reg[reg_features]
    y_reg = df_reg[reg_target]

    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )

    scaler_reg = StandardScaler()
    Xr_train_s = scaler_reg.fit_transform(Xr_train)
    Xr_test_s = scaler_reg.transform(Xr_test)

    # linear regression
    lin_reg = LinearRegression()
    lin_reg.fit(Xr_train_s, yr_train)
    yr_pred_lin = lin_reg.predict(Xr_test_s)

    # random forest reg
    rf_reg = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        random_state=42
    )
    rf_reg.fit(Xr_train_s, yr_train)
    yr_pred_rf = rf_reg.predict(Xr_test_s)

    results["regression"] = {
        "scaler": scaler_reg,
        "X_test": Xr_test,
        "X_test_scaled": Xr_test_s,
        "y_test": yr_test,
        "lin_reg": lin_reg,
        "rf_reg": rf_reg,
        "pred_lin": yr_pred_lin,
        "pred_rf": yr_pred_rf,
    }

    # ---------- CLUSTERING ----------
    if "2020_census_tract" in df.columns:
        ct_col = "2020_census_tract"
    elif "census_tract_2020" in df.columns:
        ct_col = "census_tract_2020"
    else:
        ct_col = None

    cl_features = ["model_year", "electric_range"]
    if ct_col:
        cl_features.append(ct_col)

    df_km = df[cl_features].dropna().copy()

    scaler_km = StandardScaler()
    X_km = scaler_km.fit_transform(df_km)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_km)
    df_km["cluster"] = clusters

    results["clustering"] = {
        "df_km": df_km,
        "features": cl_features,
        "scaler": scaler_km,
        "kmeans": kmeans,
    }

    return results


# ------------------------------------------------
# 3. DASHBOARD LAYOUT
# ------------------------------------------------

df = load_and_clean_data(DATA_PATH)
models = train_models(df)

st.title("🚗 EV Predictive Analysis Dashboard")
st.write("Dataset size after cleaning & sampling:", df.shape)

page = st.sidebar.radio(
    "Select section",
    ["Overview / EDA", "Classification (CAFV)", "Regression (Range)", "Clustering (K-Means)"]
)

# ---------- OVERVIEW ----------
if page == "Overview / EDA":
    st.header("Data Overview & EDA")

    st.subheader("Sample of Data")
    st.dataframe(df.head())

    st.subheader("Numeric Summary")
    st.write(df.describe())

    st.subheader("CAFV Eligibility Distribution")
    fig, ax = plt.subplots(figsize=(6, 4))
    df["clean_alternative_fuel_vehicle_cafv_eligibility"].value_counts().plot(
        kind="bar", ax=ax
    )
    ax.set_xlabel("CAFV Eligibility")
    ax.set_ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)

    st.subheader("Electric Range Distribution")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(df["electric_range"], bins=40, ax=ax)
    ax.set_xlabel("Electric Range (miles)")
    st.pyplot(fig)

    st.subheader("Model Year vs Electric Range")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(df["model_year"], df["electric_range"], alpha=0.3)
    ax.set_xlabel("Model Year")
    ax.set_ylabel("Electric Range")
    st.pyplot(fig)

# ---------- CLASSIFICATION ----------
elif page == "Classification (CAFV)":
    st.header("Classification – CAFV Eligibility")

    clf = models["classification"]
    X_test = clf["X_test"]
    y_test = clf["y_test"]
    pred_log = clf["pred_log"]
    pred_rf = clf["pred_rf"]
    proba_rf = clf["proba_rf"]

    st.subheader("Model Metrics (Logistic vs Random Forest)")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Logistic Regression")
        acc_log = accuracy_score(y_test, pred_log)
        st.write("Accuracy:", round(acc_log, 3))
        st.text("Classification Report:")
        st.text(classification_report(y_test, pred_log))

    with col2:
        st.markdown("### Random Forest Classifier")
        acc_rf = accuracy_score(y_test, pred_rf)
        st.write("Accuracy:", round(acc_rf, 3))
        st.text("Classification Report:")
        st.text(classification_report(y_test, pred_rf))

    # ===== CLASSIFICATION COMPARISON TABLE =====
    st.subheader("🔍 Classification Model Comparison (Table)")

    rows = []
    for name, preds in {
        "Logistic Regression": pred_log,
        "Random Forest Classifier": pred_rf
    }.items():
        acc = accuracy_score(y_test, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, preds, average="weighted", zero_division=0
        )
        rows.append({
            "Model": name,
            "Accuracy": acc,
            "Precision (weighted)": prec,
            "Recall (weighted)": rec,
            "F1-score (weighted)": f1
        })

    clf_comparison_df = pd.DataFrame(rows)
    st.dataframe(clf_comparison_df.style.format({
        "Accuracy": "{:.3f}",
        "Precision (weighted)": "{:.3f}",
        "Recall (weighted)": "{:.3f}",
        "F1-score (weighted)": "{:.3f}",
    }))

    # ===== CLASSIFICATION COMPARISON PLOT =====
    st.subheader("📊 Classification Comparison Plot")

    metrics_to_plot = ["Accuracy", "Precision (weighted)", "Recall (weighted)", "F1-score (weighted)"]
    fig, ax = plt.subplots(figsize=(8, 5))

    for metric in metrics_to_plot:
        ax.plot(clf_comparison_df["Model"], clf_comparison_df[metric],
                marker='o', linewidth=2, label=metric)

    ax.set_title("Classification Model Comparison")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig)

    # Confusion matrix for RF
    st.subheader("Confusion Matrix – Random Forest")
    cm = confusion_matrix(y_test, pred_rf)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=clf["rf_clf"].classes_,
        yticklabels=clf["rf_clf"].classes_,
        ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    st.pyplot(fig)

    # ROC curve (only if binary)
    if len(clf["rf_clf"].classes_) == 2 and proba_rf is not None:
        st.subheader("ROC Curve – Random Forest")
        # encode labels as 0/1
        y_true_bin = (y_test == clf["rf_clf"].classes_[1]).astype(int)
        fpr, tpr, _ = roc_curve(y_true_bin, proba_rf)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        ax.plot([0, 1], [0, 1], "k--")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend(loc="lower right")
        st.pyplot(fig)

# ---------- REGRESSION ----------
elif page == "Regression (Range)":
    st.header("Regression – Predict Electric Range")

    reg = models["regression"]
    y_test = reg["y_test"]
    pred_lin = reg["pred_lin"]
    pred_rf = reg["pred_rf"]

    # compute metrics for both models
    mse_lin = mean_squared_error(y_test, pred_lin)
    rmse_lin = np.sqrt(mse_lin)
    r2_lin = r2_score(y_test, pred_lin)

    mse_rf = mean_squared_error(y_test, pred_rf)
    rmse_rf = np.sqrt(mse_rf)
    r2_rf = r2_score(y_test, pred_rf)

    col1, col2 = st.columns(2)

    # Linear Regression metrics
    with col1:
        st.markdown("### Linear Regression (baseline)")
        st.write("MSE :", round(mse_lin, 2))
        st.write("RMSE:", round(rmse_lin, 2))
        st.write("R²  :", round(r2_lin, 3))

    # Random Forest Regression metrics
    with col2:
        st.markdown("### Random Forest Regression")
        st.write("MSE :", round(mse_rf, 2))
        st.write("RMSE:", round(rmse_rf, 2))
        st.write("R²  :", round(r2_rf, 3))

    # ===== REGRESSION COMPARISON TABLE =====
    st.subheader("🔍 Regression Model Comparison (Table)")

    reg_comparison_df = pd.DataFrame([
        {
            "Model": "Linear Regression",
            "MSE": mse_lin,
            "RMSE": rmse_lin,
            "R²": r2_lin
        },
        {
            "Model": "Random Forest Regression",
            "MSE": mse_rf,
            "RMSE": rmse_rf,
            "R²": r2_rf
        }
    ])

    st.dataframe(reg_comparison_df.style.format({
        "MSE": "{:.2f}",
        "RMSE": "{:.2f}",
        "R²": "{:.3f}",
    }))

    # ===== REGRESSION COMPARISON PLOTS =====
    st.subheader("📊 Regression Comparison Plots")

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # RMSE
    ax[0].bar(reg_comparison_df["Model"], reg_comparison_df["RMSE"], color=["skyblue", "orange"])
    ax[0].set_title("Regression RMSE Comparison")
    ax[0].set_ylabel("RMSE (Lower is better)")
    ax[0].grid(True, alpha=0.3)

    # R²
    ax[1].bar(reg_comparison_df["Model"], reg_comparison_df["R²"], color=["skyblue", "orange"])
    ax[1].set_title("Regression R² Comparison")
    ax[1].set_ylabel("R² Score (Higher is better)")
    ax[1].set_ylim(0, 1)
    ax[1].grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)

    # scatter actual vs predicted RF
    st.subheader("Random Forest – Actual vs Predicted Range")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_test, pred_rf, alpha=0.3)
    line_min = min(y_test.min(), pred_rf.min())
    line_max = max(y_test.max(), pred_rf.max())
    ax.plot([line_min, line_max], [line_min, line_max], "r--", label="Perfect prediction")
    ax.set_xlabel("Actual Range")
    ax.set_ylabel("Predicted Range")
    ax.legend()
    st.pyplot(fig)

# ---------- CLUSTERING ----------
elif page == "Clustering (K-Means)":
    st.header("K-Means Clustering – EV Groups")

    cl = models["clustering"]
    df_km = cl["df_km"]
    features = cl["features"]

    st.write("Using features for clustering:", features)

    st.subheader("Cluster Sizes")
    st.write(df_km["cluster"].value_counts())

    st.subheader("Cluster-wise Means")
    st.write(df_km.groupby("cluster").mean(numeric_only=True))

    st.subheader("Scatter: Model Year vs Electric Range")
    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(df_km["model_year"], df_km["electric_range"],
                         c=df_km["cluster"], cmap="viridis", alpha=0.5)
    ax.set_xlabel("Model Year")
    ax.set_ylabel("Electric Range")
    legend1 = ax.legend(*scatter.legend_elements(),
                        title="Cluster", loc="upper left")
    ax.add_artist(legend1)
    st.pyplot(fig)
