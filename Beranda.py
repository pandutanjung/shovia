import streamlit as st

# ==========================================
# KONFIGURASI HALAMAN (Harus di baris pertama)
# ==========================================
st.set_page_config(
    page_title="Shovia",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# KONTEN BERANDA
# ==========================================
st.title("Shovia 🎥📊")
st.subheader("Short Video Analytics")

st.markdown("""
---
**Shovia** adalah *dashboard* analitik interaktif yang dirancang untuk melakukan ekstraksi dan klasifikasi sentimen opini publik pada tayangan video pendek secara otomatis. 

Sistem ini ditenagai oleh integrasi API Apify untuk *scraping* data dan model *Machine Learning* **Naive Bayes** untuk analisis sentimen.

### 🚀 Alur Penggunaan Aplikasi:

Aplikasi ini terbagi menjadi dua modul utama yang dapat diakses melalui **sidebar di sebelah kiri**:

1. **Konfigurasi**
   * Masukkan *username* atau *link* profil Instagram target.
   * Sistem akan melakukan *scraping* data video sekaligus mengekstrak transkripsi audionya.
   * Anda dapat melakukan *filtering* data berdasarkan *keyword* pada *caption*.
   * Pilih data spesifik (sebagian atau seluruhnya) untuk diproses oleh model klasifikasi sentimen.

2. **Visualisasi**
   * Menampilkan hasil akhir klasifikasi sentimen dalam format tabel komprehensif (termasuk kolom nomor urut).
   * Menyajikan *Pie Chart* interaktif untuk menganalisis proporsi distribusi sentimen (Positif dan Negatif) dari video yang telah dianalisis.
   * Menampilkan statistik *engagement* secara akumulatif (total *Views*, *Likes*, dan *Comments*) dari data hasil analisis.
   * Memvisualisasikan analisis *N-Gram* (Unigram, Bigram, Trigram) untuk memetakan kata atau frasa yang paling sering muncul pada masing-masing kelas sentimen, dilengkapi fitur pengaturan *stopwords* interaktif.

---
👈 **Silakan klik menu `Konfigurasi` di sidebar sebelah kiri untuk memulai proses analisis.**
""")

# ==========================================
# INISIALISASI SESSION STATE (Opsional tapi direkomendasikan di awal)
# ==========================================
# Kita siapkan 'wadah' kosong di memori untuk menyimpan data antar halaman
if 'df_hasil_klasifikasi' not in st.session_state:
    st.session_state['df_hasil_klasifikasi'] = None