import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Prediksi Jumlah Pemesanan Produk",
    page_icon="📦",
    layout="wide"
)

# --- 2. FUNGSI PEMUATAN DATA & PELATIHAN MODEL ---
@st.cache_data
def load_and_train():
    DATA_PATH = 'data_penjualan.csv'  # Sesuaikan path lokal Anda
    try:
        df = pd.read_csv(DATA_PATH, sep=';')
    except FileNotFoundError:
        st.error(f"❌ File tidak ditemukan di path: '{DATA_PATH}'")
        st.stop()

    # Pembersihan Data
    for col in ['Harga', 'Total', 'Jumlah Order']:
        if df[col].dtype == 'object':
            df[col] = df[col].str.replace(r'[^\d]', '', regex=True).astype(float)

    df.dropna(inplace=True)

    # One-Hot Encoding untuk fitur kategorikal
    df_encoded = pd.get_dummies(df, columns=['Jenis Produk'], drop_first=True)

    # Definisi fitur (X) dan target (y)
    y = df_encoded['Jumlah Order']
    X = df_encoded.drop(columns=['Tanggal', 'Jumlah Order', 'Total'])

    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model Regresi Linear
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    unique_products = df['Jenis Produk'].unique().tolist()

    return model, mse, unique_products, feature_names, df, model.coef_

# Memuat data dan melatih model
model, mse, unique_products, feature_names, df, coefficients = load_and_train()

# --- 3. TAMPILAN ANTARMUKA (UI) ---
st.title("📦 Prediksi Jumlah Pemesanan Produk Kemasan Berdasarkan Harga dan Jenis Produk")

# Penjelasan model
st.header("🧠 Tentang Model Prediksi")
st.markdown("""
Model yang digunakan untuk memprediksi jumlah pemesanan produk kemasan adalah **Regresi Linear (Linear Regression)**.  
Model ini bekerja dengan mencari hubungan linier antara fitur-fitur yang digunakan (seperti **harga** dan **jenis produk**) terhadap target yang diprediksi, yaitu **jumlah pemesanan**.

Setiap faktor akan diberikan bobot (koefisien) yang menunjukkan seberapa besar pengaruhnya terhadap jumlah pesanan.  
Semakin besar nilai koefisien suatu fitur, semakin besar pula kontribusinya dalam menghasilkan prediksi jumlah pesanan.
""")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "✍️ Prediksi Jumlah Order",
    "📊 Evaluasi Model",
    "🤖 Faktor Penting (Menurut Model)",
    "🔍 Analisis Data Awal"
])

# Tab 1: Prediksi
with tab1:
    st.header("Prediksi Jumlah Order")
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            selected_product = st.selectbox("Pilih Jenis Produk", options=unique_products)
        with col2:
            price_input = st.slider(
                "Harga per Item",
                min_value=int(df['Harga'].min()),
                max_value=int(df['Harga'].max()),
                value=int(df['Harga'].median()),
                step=1000
            )

        submitted = st.form_submit_button("Prediksi Jumlah Order")

        if submitted:
            input_df = pd.DataFrame(columns=feature_names)
            input_df.loc[0] = 0
            input_df['Harga'] = price_input

            product_column_name = f"Jenis Produk_{selected_product}"
            if product_column_name in input_df.columns:
                input_df[product_column_name] = 1

            prediction = model.predict(input_df)[0]
            predicted_quantity = round(max(0, prediction))

            st.success(f"### 🛍️ Prediksi Jumlah Order: **{predicted_quantity}** item")
            st.info(
                f"Untuk produk **{selected_product}** dengan harga **Rp {price_input:,}**, "
                f"model memprediksi akan terjual sekitar **{predicted_quantity}** item."
            )

# Tab 2: Evaluasi Model
with tab2:
    st.header("📊 Evaluasi Model Regresi Linear")
    st.metric(label="Mean Squared Error (MSE)", value=round(mse, 2))
    st.info("MSE mengukur rata-rata kesalahan prediksi. Semakin kecil nilainya, semakin baik performa model.")

# Tab 3: Faktor Penting
with tab3:
    st.header("🤖 Faktor Paling Berpengaruh")
    coef_df = pd.DataFrame({
        'Faktor': feature_names,
        'Koefisien': coefficients
    }).sort_values(by='Koefisien', ascending=False)

    st.subheader("📌 Kontribusi Masing-Masing Faktor")
    st.dataframe(coef_df, use_container_width=True, hide_index=True)

# Tab 4: Analisis Data Awal
with tab4:
    st.header("🔍 Analisis Data Berdasarkan Rata-rata & Korelasi")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📦 Rata-rata Jumlah Order per Jenis Produk")
        pengaruh_produk = df.groupby('Jenis Produk')['Jumlah Order'].mean().sort_values(ascending=False).reset_index()
        pengaruh_produk.columns = ['Jenis Produk', 'Rata-rata Jumlah Order']
        st.dataframe(pengaruh_produk, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("#### 💸 Korelasi Harga vs Jumlah Order")
        korelasi_harga = df['Harga'].corr(df['Jumlah Order'])
        st.metric(label="Koefisien Korelasi", value=round(korelasi_harga, 2))

        if korelasi_harga < -0.5:
            st.warning("Semakin tinggi harga, semakin rendah jumlah pesanan. Korelasi negatif kuat.")
        elif korelasi_harga < 0:
            st.warning("Ada kecenderungan negatif. Harga naik → pesanan turun.")
        else:
            st.success("Hubungan tidak terlalu kuat. Harga bukan satu-satunya faktor.")
