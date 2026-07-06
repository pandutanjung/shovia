import streamlit as st
import pandas as pd
from apify_client import ApifyClient
import re
import joblib

# ==========================================
# KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(page_title="Shovia - Konfigurasi", page_icon="🎥", layout="wide")
st.title("🔍 Scraping & Transkripsi")

# ==========================================
# 1. FORM INPUT APIFY
# ==========================================
with st.expander("⚙️ Pengaturan", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        api_token = st.text_input("Apify API Token", type="password", help="Masukkan API Token dari akun Apify Anda.")
        username = st.text_input("Username", placeholder="Contoh: natgeo")
    with col2:
        results_limit = st.number_input("Maksimum reels per profile", min_value=1, max_value=1000, value=27)
        only_newer_than = st.date_input("Batas tanggal terbaru (Opsional)", value=None, help="Kosongkan jika ingin mengambil semua tanggal.")

btn_scrape = st.button("🚀 Mulai", type="primary")

# ==========================================
# 2. PROSES SCRAPING
# ==========================================
if btn_scrape:
    if not api_token or not username:
        st.warning("⚠️ API Token dan Username wajib diisi!")
    else:
        with st.spinner("Sedang menghubungi server Apify... Proses transkripsi mungkin memakan waktu beberapa menit."):
            try:
                client = ApifyClient(api_token)

                run_input = {
                    "username": [username],
                    "resultsLimit": results_limit,
                    "skipPinnedPosts": False,
                    "includeSharesCount": True,
                    "includeTranscript": True,
                    "includeDownloadedVideo": False,
                }

                if only_newer_than:
                    if isinstance(only_newer_than, tuple):
                        if len(only_newer_than) > 0:
                            run_input["onlyPostsNewerThan"] = only_newer_than[0].strftime("%Y-%m-%d")
                    else:
                        run_input["onlyPostsNewerThan"] = only_newer_than.strftime("%Y-%m-%d")

                run = client.actor("xMc5Ga1oCONPmWJIa").call(run_input=run_input)

                results = []
                for item in client.dataset(run["defaultDatasetId"]).iterate_items():
                    results.append(item)

                if results:
                    df = pd.DataFrame(results)

                    kolom_target = ['caption', 'commentsCount', 'likesCount', 'ownerUsername',
                                    'timestamp', 'transcript', 'url', 'videoViewCount']

                    for col in kolom_target:
                        if col not in df.columns:
                            df[col] = None

                    st.session_state['df_raw'] = df[kolom_target]
                    st.success(f"✅ Scraping berhasil! Ditemukan {len(df)} data video.")
                else:
                    st.error("Tidak ada data video yang ditemukan dari akun tersebut.")

            except Exception as e:
                st.error(f"❌ Terjadi kesalahan pada Apify: {e}")

# ==========================================
# 3. FILTERING & SELEKSI DATA
# ==========================================
if 'df_raw' in st.session_state and st.session_state['df_raw'] is not None:
    st.divider()
    st.subheader("📋 Hasil Scraping & Seleksi Data")

    df_display = st.session_state['df_raw'].copy()

    keyword = st.text_input("🔍 Filter berdasarkan kata kunci pada caption (Opsional):", placeholder="Ketik kata kunci...")
    if keyword:
        df_display = df_display[df_display['caption'].fillna('').str.contains(keyword, case=False)]
        st.caption(f"Ditemukan {len(df_display)} data yang mengandung kata '{keyword}'.")

    df_display.insert(0, "Pilih", True)

    st.write("Silakan centang pada kolom **'Pilih'** untuk menentukan data mana saja yang akan diteruskan ke proses Klasifikasi Sentimen.")

    edited_df = st.data_editor(
        df_display,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Pilih": st.column_config.CheckboxColumn("Pilih Data", default=True)
        }
    )

    if st.button("Lanjutkan ke Klasifikasi Sentimen ➡️", type="primary"):
        df_selected = edited_df[edited_df["Pilih"] == True].copy()
        df_selected = df_selected.drop(columns=["Pilih"])
        st.session_state['df_to_classify'] = df_selected
        st.success(f"✅ {len(df_selected)} Data berhasil dikunci untuk diklasifikasi!")

# ==========================================
# 4. FUNGSI PRAPROSES TEKS
# ==========================================
def light_clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-z0-9\s.,!?]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ==========================================
# 5. KLASIFIKASI SENTIMEN — NAIVE BAYES
# ==========================================
if 'df_to_classify' in st.session_state and not st.session_state['df_to_classify'].empty:
    st.divider()
    st.subheader("🤖 Klasifikasi Sentimen")
    st.info("Metode klasifikasi yang digunakan: **Naive Bayes**")

    if st.button("⚙️ Jalankan Proses Klasifikasi", type="primary"):
        df_final = st.session_state['df_to_classify'].copy()

        with st.spinner("Sedang membersihkan teks..."):
            df_final['clean_transcript'] = df_final['transcript'].apply(light_clean_text)

        with st.spinner("Sedang mengklasifikasikan menggunakan Naive Bayes..."):
            try:
                vectorizer_path = "E:\\shovia\\tfidf_vectorizer.pkl"
                model_path = "E:\\shovia\\model_nb_deepseek_imbalanced.pkl"

                tfidf = joblib.load(vectorizer_path)
                model_nb = joblib.load(model_path)

                X_tfidf = tfidf.transform(df_final['clean_transcript'])
                df_final['sentimen'] = model_nb.predict(X_tfidf)

            except FileNotFoundError:
                st.error("❌ File model atau vectorizer tidak ditemukan. Pastikan path sudah benar.")
                st.stop()

        st.session_state['df_final_classified'] = df_final
        st.success("✅ Proses Klasifikasi berhasil diselesaikan!")
        st.info("ℹ️ Silakan navigasi ke menu **'Visualisasi'** di sidebar sebelah kiri.")