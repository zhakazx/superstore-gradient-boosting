import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ===============================
# Konfigurasi Halaman
# ===============================
st.set_page_config(page_title="Superstore ML Auto", layout="wide")
st.title("📊 Analisis Profitabilitas Superstore")

# ===============================
# KONFIGURASI HYPERPARAMETER
# ===============================
TEST_SIZE = 0.2
N_ESTIMATORS = 100
LEARNING_RATE = 0.1
MAX_DEPTH = 7
RANDOM_STATE = 42


# ===============================
# Load Data
# ===============================
@st.cache_data
def load_data():
    try:
        return pd.read_csv("Sample - Superstore.csv", encoding="ISO-8859-1")
    except FileNotFoundError:
        st.error(
            "File 'Sample - Superstore.csv' tidak ditemukan. Pastikan file berada di folder yang sama."
        )
        return None


df = load_data()
rows, cols = df.shape

col_i1, col_i2 = st.columns(2)
col_i1.metric("Jumlah Baris", rows)
col_i2.metric("Jumlah Kolom", cols)

if df is not None:
    # ===============================
    # OUTLIER REMOVAL
    # ===============================
    # Hapus 1% data teratas dan 1% data terbawah (Outlier ekstrem)
    q_low = df["Profit"].quantile(0.01)
    q_hi = df["Profit"].quantile(0.99)
    df_filtered = df[(df["Profit"] > q_low) & (df["Profit"] < q_hi)]
    df = df_filtered

    with st.expander("🔍 Lihat Sampel Data Awal"):
        st.dataframe(df.head())

    st.write("---")
    st.info("🚀 Memulai proses pelatihan model otomatis...")

    # ===============================
    # Feature Engineering (Dipakai untuk Keduanya)
    # ===============================
    with st.spinner("Sedang melakukan Feature Engineering..."):
        df_model = df.copy()

        # Fitur Tambahan
        df_model["Unit_Price"] = df_model["Sales"] / df_model["Quantity"]
        df_model["Discount_Amount"] = df_model["Sales"] * df_model["Discount"]
        df_model["Sales_Log"] = np.log1p(df_model["Sales"])

        features = [
            "Ship Mode",
            "Segment",
            "Region",
            "Category",
            "Sub-Category",
            "Sales",
            "Quantity",
            "Discount",
            "Unit_Price",
            "Discount_Amount",
            "Sales_Log",
        ]

        X = df_model[features].copy()

        # Label Encoding
        cat_cols = X.select_dtypes(include="object").columns
        for col in cat_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])

    # ===============================
    # BAGIAN 1: KLASIFIKASI
    # ===============================
    st.header("1️⃣ Hasil Klasifikasi (Untung vs Rugi)")
    st.info("Target: Memprediksi apakah transaksi Profitable (1) atau Tidak (0).")

    # Target Klasifikasi: 1 jika Profit > 0, 0 jika tidak
    df_model["is_profitable"] = (df_model["Profit"] > 0).astype(int)
    y_class = df_model["is_profitable"]

    # Split Data Klasifikasi
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X, y_class, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_class
    )

    # Train Model Klasifikasi
    clf = GradientBoostingClassifier(
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        max_depth=MAX_DEPTH,
        random_state=RANDOM_STATE,
    )
    clf.fit(X_train_c, y_train_c)
    preds_c = clf.predict(X_test_c)

    # Tampilkan Metrik Klasifikasi
    col_c1, col_c2 = st.columns(2)
    acc = accuracy_score(y_test_c, preds_c)
    col_c1.metric("Akurasi Model", f"{acc:.2%}")

    with col_c2:
        st.text("Classification Report:")
        st.code(classification_report(y_test_c, preds_c, output_dict=False))

    # Plot Confusion Matrix
    cm = confusion_matrix(y_test_c, preds_c)
    fig_cm, ax_cm = plt.subplots(figsize=(6, 3))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Rugi (0)", "Untung (1)"],
        yticklabels=["Rugi (0)", "Untung (1)"],
        ax=ax_cm,
    )
    ax_cm.set_title("Confusion Matrix")
    st.pyplot(fig_cm)

    st.write("---")

    # ===============================
    # BAGIAN 2: REGRESI
    # ===============================
    st.header("2️⃣ Hasil Regresi (Prediksi Nilai Profit)")
    st.info("Target: Memprediksi nilai nominal Profit secara spesifik.")

    # Target Regresi: Nilai Profit Asli
    y_reg = df_model["Profit"]

    # Split Data Regresi
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X, y_reg, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Train Model Regresi
    reg = GradientBoostingRegressor(
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        max_depth=MAX_DEPTH,
        random_state=RANDOM_STATE,
    )
    reg.fit(X_train_r, y_train_r)
    preds_r = reg.predict(X_test_r)

    # Tampilkan Metrik Regresi
    mse = mean_squared_error(y_test_r, preds_r)
    rmse = mse**0.5
    r2 = r2_score(y_test_r, preds_r)

    col_r1, col_r2 = st.columns(2)
    col_r1.metric("RMSE (Error dalam $)", f"${rmse:.2f}")
    col_r2.metric("R² Score", f"{r2:.4f}")

    # Plot Scatter Regresi
    fig_reg, ax_reg = plt.subplots(figsize=(8, 4))
    sns.scatterplot(x=y_test_r, y=preds_r, alpha=0.5, color="green", ax=ax_reg)

    # Garis diagonal perfect prediction
    min_val = min(y_test_r.min(), preds_r.min())
    max_val = max(y_test_r.max(), preds_r.max())
    ax_reg.plot(
        [min_val, max_val], [min_val, max_val], color="red", linestyle="--", linewidth=2
    )

    ax_reg.set_xlabel("Profit Aktual")
    ax_reg.set_ylabel("Profit Prediksi")
    ax_reg.set_title("Scatter Plot: Aktual vs Prediksi")
    st.pyplot(fig_reg)

    # Tabel Sampel
    with st.expander("📈 Lihat Tabel Detail Prediksi (Regresi)"):
        result_df = pd.DataFrame(
            {
                "Actual Profit": y_test_r.values[:15],
                "Predicted Profit": preds_r[:15],
                "Selisih": y_test_r.values[:15] - preds_r[:15],
            }
        )
        st.dataframe(result_df.style.format("{:.2f}"))

else:
    st.warning("Menunggu file 'Sample - Superstore.csv'...")
