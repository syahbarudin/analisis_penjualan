import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Prediksi Penjualan",
    page_icon="📈",
    layout="wide" # Mengubah layout menjadi wide agar lebih lega
)

# --- 2. FUNGSI PEMUATAN DATA & PELATIHAN MODEL ---
@st.cache_data
def load_and_train():
    DATA_PATH = 'F:\Download\Data\data_penjualan.csv' # Pastikan nama file sesuai
    try:
        df = pd.read_csv(DATA_PATH, sep=';')
    except FileNotFoundError:
        st.error(f"❌ File tidak ditemukan di path: '{DATA_PATH}'")
        st.stop()

    # PEMBERSIHAN DATA
    for col in ['Harga', 'Total', 'Jumlah Order']:
        if df[col].dtype == 'object':
            df[col] = df[col].str.replace(r'[^\d]', '', regex=True).astype(float)
    
    df.dropna(inplace=True)

    # ONE-HOT ENCODING
    df_encoded = pd.get_dummies(df, columns=['Jenis Produk'], drop_first=True)

    # DEFINISI FITUR (X) DAN TARGET (y)
    y = df_encoded['Jumlah Order']
    X = df_encoded.drop(columns=['Tanggal', 'Jumlah Order', 'Total'])

    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, np.round(y_pred))
    
    unique_products = df['Jenis Produk'].unique().tolist()
    
    # Kita return df asli (sebelum encoding) untuk analisis deskriptif
    return model, mse, unique_products, feature_names, df

# Memuat data dan melatih model
model, mse, unique_products, feature_names, df = load_and_train()


# --- 3. TAMPILAN ANTARMUKA (UI) ---
st.title("📈 Analisis & Prediksi Penjualan Fast Food")

# PENAMBAHAN TAB KE-4
tab1, tab2, tab3, tab4 = st.tabs([
    "✍️ Prediksi Jumlah Order", 
    "📊 Evaluasi Model", 
    "🤖 Faktor Penting (Menurut Model)",
    "🔍 Analisis Data Awal"
])

with tab1:
    st.header("Prediksi Jumlah Order")
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            selected_product = st.selectbox("Pilih Jenis Produk", options=unique_products)
        with col2:
            price_input = st.slider("Harga per Item", min_value=int(df['Harga'].min()), max_value=int(df['Harga'].max()), value=int(df['Harga'].median()), step=1000)
        
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
            st.info(f"Untuk produk **{selected_product}** dengan harga **Rp {price_input:,}**, model memprediksi akan terjual sekitar **{predicted_quantity}** item.")

with tab2:
    st.header("Evaluasi Kinerja Model")
    st.metric(label="Mean Squared Error (MSE)", value=round(mse, 2))
    st.info("MSE mengukur rata-rata kesalahan prediksi model. Semakin kecil nilainya, semakin akurat model dalam memprediksi jumlah order.")

with tab3:
    st.header("Faktor Paling Penting Menurut Model")
    st.info("Analisis ini menunjukkan fitur mana yang paling diandalkan oleh model **Random Forest** untuk membuat prediksi. Ini membantu memahami 'pikiran' model.")
    
    importances = pd.DataFrame({
        'Faktor': feature_names,
        'Tingkat Kepentingan': model.feature_importances_
    }).sort_values(by='Tingkat Kepentingan', ascending=False)
    
    st.dataframe(importances, use_container_width=True, hide_index=True)

# --- KODE BARU UNTUK ANALISIS BERDASARKAN RATA-RATA ---
with tab4:
    st.header("Analisis Faktor Pengaruh (Berdasarkan Data Rata-rata & Korelasi)")
    st.info("""
    Analisis ini berbeda dengan analisis model. Di sini kita melihat tren langsung dari data mentah, 
    tanpa menggunakan machine learning, untuk mendapatkan intuisi bisnis dasar.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### **Pengaruh Jenis Produk**")
        st.markdown("Menunjukkan produk mana yang rata-rata paling banyak dipesan.")
        
        # Menghitung rata-rata Jumlah Order per Jenis Produk
        pengaruh_produk = df.groupby('Jenis Produk')['Jumlah Order'].mean().sort_values(ascending=False).reset_index()
        pengaruh_produk.columns = ['Jenis Produk', 'Rata-rata Jumlah Order']
        
        st.dataframe(
            pengaruh_produk,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Rata-rata Jumlah Order": st.column_config.ProgressColumn(
                    "Rata-rata Jumlah Order",
                    format="%.2f",
                    min_value=0,
                    max_value=float(pengaruh_produk['Rata-rata Jumlah Order'].max()),
                ),
            }
        )

    with col2:
        st.markdown("#### **Pengaruh Harga**")
        st.markdown("Mengukur hubungan linear antara harga dan jumlah pesanan.")

        # Menghitung korelasi
        korelasi_harga = df['Harga'].corr(df['Jumlah Order'])

        st.metric(label="Koefisien Korelasi Harga vs Jumlah Order", value=round(korelasi_harga, 2))
        
        if korelasi_harga < -0.5:
            st.warning("Artinya: Ada hubungan terbalik yang kuat. **Semakin tinggi harga, semakin rendah jumlah pesanan.** Ini adalah tren yang diharapkan.")
        elif korelasi_harga < 0:
            st.warning("Artinya: Ada hubungan terbalik. Semakin tinggi harga, cenderung semakin rendah jumlah pesanan.")
        else:
            st.success("Artinya: Hubungan terbalik tidak kuat atau bahkan positif. Ini mungkin menunjukkan bahwa harga bukan satu-satunya penentu utama.")
        
        st.markdown("""
        **Cara membaca korelasi:**
        - **-1**: Hubungan terbalik sempurna.
        - **0**: Tidak ada hubungan linear.
        - **+1**: Hubungan lurus sempurna.
        """)