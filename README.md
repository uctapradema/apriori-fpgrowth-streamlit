# 🛒 Analisis Sensitivitas Apriori dan FP-Growth

Aplikasi Streamlit untuk menganalisis pengaruh **minimum support**, **minimum confidence**, dan **granularitas item** terhadap keluaran dan biaya komputasi algoritma **Apriori** dan **FP-Growth** pada data transaksi ritel, lengkap dengan **seleksi aturan robust** (uji signifikansi + stabilitas) dan rekomendasi promosi.

Dikembangkan untuk Penelitian Pemula LPPM Universitas Ngudi Waluyo 2026:
*"Sensitivitas Parameter dan Granularitas Item Apriori dan FP-Growth pada Data Transaksi Ritel"*.

## Fitur

| Tab | Isi |
|---|---|
| ① Data & Pembersihan | Deteksi format log POS / CSV umum, aturan pembersihan, pemeriksaan TTL vs PLU, unduh data bersih |
| ② Profil Data | Kepadatan matriks, ukuran keranjang, sebaran support item, editor nama departemen |
| ③ Sensitivitas Parameter | Grid support × confidence, cek kesamaan keluaran Apriori vs FP-Growth, heatmap, kurva respons, tabel ΔI |
| ④ Biaya Komputasi | Waktu median (n ulangan) dan memori puncak, deteksi titik silang |
| ⑤ Aturan & Uji Robust | Uji Fisher + Benjamini–Hochberg, uji dua paruh periode, rekurensi subsampel |
| ⑥ Rekomendasi & Unduh | Kalimat rekomendasi promosi, unduh CSV/Excel/JSON |

## Struktur repositori

```
.
├── app.py                  # aplikasi Streamlit (UI)
├── PANDUAN.md              # panduan yang tampil di tab ℹ️
├── requirements.txt
├── .streamlit/config.toml  # tema & batas unggah
├── data/
│   └── contoh_transaksi.csv  # data SINTETIS untuk demo
├── src/
│   ├── data.py       # baca CSV, pembersihan log POS / format umum, granularitas
│   ├── mining.py     # matriks transaksi-item, Apriori/FP-Growth, grid, ΔI
│   ├── robust.py     # Fisher + BH, uji dua paruh, rekurensi subsampel
│   ├── recommend.py  # pelabelan aturan & teks rekomendasi
│   └── viz.py        # grafik Plotly
└── tests/
    └── test_pipeline.py
```

## Menjalankan di komputer sendiri

Butuh Python 3.10–3.12.

```bash
git clone https://github.com/<akun-anda>/<nama-repo>.git
cd <nama-repo>
python -m venv .venv
# Windows: .venv\Scripts\activate     | macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Buka `http://localhost:8501` di browser.

Uji cepat (opsional): `pip install pytest && pytest -q`

## Unggah ke GitHub

```bash
git init
git add .
git commit -m "Aplikasi analisis sensitivitas Apriori vs FP-Growth"
git branch -M main
git remote add origin https://github.com/<akun-anda>/<nama-repo>.git
git push -u origin main
```

Atau lewat web: buat repositori baru di github.com → **Add file → Upload files** → seret seluruh isi folder ini (termasuk folder `src`, `data`, `.streamlit`).

> ⚠️ **Jangan unggah data transaksi asli toko ke repositori publik.** File `.gitignore` sudah mengabaikan `data/*.csv` kecuali `contoh_transaksi.csv`. Unggah data asli langsung lewat tombol upload di aplikasi.

## Deploy ke Streamlit Community Cloud

1. Masuk ke <https://share.streamlit.io> dengan akun GitHub.
2. **Create app → Deploy a public app from GitHub**.
3. Pilih repositori, branch `main`, *Main file path* `app.py`.
4. *Advanced settings* → Python **3.12**.
5. **Deploy**. Pemasangan paket pertama kali butuh beberapa menit.

Catatan memori: server gratis Streamlit Cloud ±1 GB RAM. Apriori mode default mlxtend pada support ≤ 0,2% (data ribuan SKU) bisa memakan >1 GB. Aktifkan **Apriori mode low_memory** atau isi **Lewati Apriori untuk support <** di pengaturan lanjutan tab ③.

## Format data

**Log POS toko** (delimiter `;`):

```
"ID";"FLAG";"CODE";"QTY";"PRICE";"REF";"DESC";"COST";"NOTA";"CHKDT";"TGL";"KASIR";"SHIFT";...
"74723";"PLU";"0230397";"2";"29000";NULL;"POPOKU DIAPERS L20";NULL;"0301013";"Y";"2015-03-01 10:39:12";"126";"1";...
```

**Format umum** (delimiter `,` / `;` / tab terdeteksi otomatis):

```
no_transaksi,kode_item,nama_item,tanggal
TRX00001,0020000,SEMBAKO VARIAN 1,2026-01-03 09:12
TRX00001,0030000,MI INSTAN VARIAN 1,2026-01-03 09:12
```

## Metode singkat

- **Algoritma**: `mlxtend` (`apriori`, `fpgrowth`, `association_rules`) dengan data dan parameter identik.
- **Waktu**: median dari *n* ulangan `time.perf_counter`; **memori**: puncak `tracemalloc` pada eksekusi terpisah.
- **Aturan robust** = q(BH) < α **dan** muncul di dua paruh periode **dan** rekurensi subsampel ≥ ambang.

## Sitasi

[PERLU DIISI: sitasi artikel setelah terbit]

## Lisensi

MIT
