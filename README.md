# 🚗 Clustering Rasio Harga vs Performa Mobil

Segmentasi mobil berdasarkan rasio harga terhadap performa (Engine HP & highway MPG) menggunakan **K-Means Clustering**, dilengkapi aplikasi **Streamlit** interaktif untuk eksplorasi data dan prediksi segmen mobil baru.

🔗 **Repo:** [FikriNash12/Clustering-Rasio-Harga-vs-Performa-Mobil](https://github.com/FikriNash12/Clustering-Rasio-Harga-vs-Performa-Mobil)

---

## 📌 Deskripsi Proyek

Proyek ini mengelompokkan mobil ke dalam tiga segmen nilai berdasarkan seberapa wajar harganya (MSRP) dibandingkan dengan performanya (Engine HP dan highway MPG):

- 🟢 **High Value** — harga lebih murah dari yang seharusnya berdasarkan performa
- 🔵 **Standard Value** — harga sesuai dengan performa
- 🔴 **Overpriced / Low Value** — harga lebih mahal dari yang seharusnya berdasarkan performa

Pendekatan yang digunakan:
1. **Regresi linear** memprediksi `log(MSRP)` dari `log(Engine HP)` dan `highway MPG`.
2. Selisih antara harga aktual dan harga prediksi dihitung sebagai fitur baru bernama **`value_gap`**.
3. **K-Means (K=3)** mengelompokkan mobil berdasarkan `Engine HP`, `highway MPG`, dan `value_gap`.

Dataset mencakup **507 model mobil (2010–2017)**, non-listrik, dengan MSRP ≤ $300.000.

---

## 📁 Struktur File

| File | Deskripsi |
|---|---|
| `50423157_Ananda_Nashril_Fikri_Bachtiar_Kelas_A_Projek_M4.ipynb` | Notebook analisis: EDA, regresi, clustering, hingga evaluasi model |
| `data.csv` | Dataset mentah mobil |
| `hasil_segmentasi_mobil.csv` | Hasil akhir data yang sudah diberi label segmen |
| `model_segmentasi_mobil.pkl` | Model terlatih (regresi, scaler, KMeans, label map) dalam satu bundle |
| `app.py` | Aplikasi Streamlit untuk eksplorasi data & prediksi segmen mobil baru |
| `requirements.txt` | Daftar dependency Python |

---

## 🖥️ Fitur Aplikasi Streamlit

1. **📊 Explorer Data** — filter data mobil berdasarkan merek, segmen, HP, MSRP, dan MPG; lihat ringkasan statistik, scatter plot, distribusi `value_gap`, dan tabel data.
2. **🔮 Prediksi Segmen** — masukkan spesifikasi mobil baru (Engine HP, highway MPG, MSRP) untuk memprediksi segmennya secara langsung.

---

## ⚙️ Cara Menjalankan Aplikasi Streamlit

### 1. Clone repository
```bash
git clone https://github.com/FikriNash12/Clustering-Rasio-Harga-vs-Performa-Mobil.git
cd Clustering-Rasio-Harga-vs-Performa-Mobil
```

### 2. (Opsional) Buat virtual environment
```bash
python -m venv venv
```
Aktifkan:
- Windows: `venv\Scripts\activate`
- Mac/Linux: `source venv/bin/activate`

### 3. Install dependency
```bash
pip install -r requirements.txt
```

### 4. Jalankan aplikasi
```bash
streamlit run app.py
```

Aplikasi akan terbuka otomatis di browser pada `http://localhost:8501`. Untuk menghentikan, tekan `Ctrl + C` di terminal.

> **Catatan:** pastikan `model_segmentasi_mobil.pkl` dan `hasil_segmentasi_mobil.csv` berada di folder yang sama dengan `app.py`.

---

## 📓 Cara Menjalankan Notebook (Google Colab)

Notebook ini dibuat dan dijalankan menggunakan **Google Colab**.

1. Buka [Google Colab](https://colab.research.google.com).
2. Pilih **File → Upload notebook**, lalu unggah file `.ipynb` dari repo ini (atau buka langsung via **File → Open notebook → GitHub**, lalu tempel URL repo).
3. Upload file `data.csv`:
   - **Cara cepat (sesi sementara):** klik ikon folder 📁 di sidebar kiri Colab → ikon upload → pilih `data.csv`. File akan hilang jika runtime direstart.
   - **Cara lebih aman:** mount Google Drive terlebih dahulu:
     ```python
     from google.colab import drive
     drive.mount('/content/drive')
     ```
     lalu simpan `data.csv` di Drive dan sesuaikan path saat `pd.read_csv(...)`.
4. Jalankan seluruh sel secara berurutan (**Runtime → Run all**, atau `Shift + Enter` per sel).
5. Jika ada library yang belum tersedia, install langsung di sel baru:
   ```python
   !pip install nama_library
   ```

---

## 🧠 Detail Model

- **Model:** K-Means (K = 3)
- **Fitur:** `Engine HP`, `highway MPG`, dan `value_gap` (residual regresi `log(MSRP)` terhadap `log(HP)` dan MPG)
- **Cakupan data:** 507 model mobil (2010–2017), non-listrik, MSRP ≤ $300.000

## ⚠️ Keterbatasan Model

Model dilatih pada data mobil pasar AS tahun 2010–2017 (harga nominal USD pada masa itu) dan hanya mencakup mobil non-listrik dengan MSRP ≤ $300.000. Prediksi untuk mobil dengan harga terkini atau pasar di luar cakupan tersebut (misalnya harga di Indonesia) **tidak divalidasi** dan sebaiknya diinterpretasikan dengan hati-hati.

---

## 🛠️ Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn (Linear Regression, KMeans, StandardScaler)
- Streamlit
- Plotly Express

---

## 👤 Author

**Ananda Nashril Fikri Bachtiar**
NPM: 50423157 — Kelas A
