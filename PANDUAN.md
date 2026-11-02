## Panduan singkat

**Alur pemakaian**

1. **Unggah data** di panel kiri. Aplikasi mengenali dua format:
   - *Log POS toko*: CSV ber-delimiter `;` dengan kolom `FLAG`, `CODE`, `DESC`, `QTY`, `PRICE`, `NOTA`, `TGL`. Pembersihan otomatis: hanya baris `PLU`, hanya nota yang punya baris `TTL`, item = `CODE`.
   - *Format umum*: CSV apa pun dengan kolom ID transaksi dan kolom item (deskripsi dan tanggal opsional).
2. **Pilih granularitas item**: SKU (kode asli), Departemen (n digit awal kode), atau Deskripsi.
3. **② Profil Data** — lihat support item tertinggi dan berapa item yang lolos tiap ambang. Rentang *minimum support* sebaiknya dipilih dari sini, bukan dari kebiasaan studi lain.
4. **③ Sensitivitas Parameter** — jalankan grid *minimum support* × *minimum confidence* untuk Apriori dan FP-Growth.
5. **④ Biaya Komputasi** — bandingkan waktu median dan memori puncak, lihat titik silang kedua algoritma.
6. **⑤ Aturan & Uji Robust** — tambang aturan pada satu konfigurasi, lalu saring dengan:
   - uji eksak Fisher satu sisi + koreksi Benjamini–Hochberg,
   - kemunculan di dua paruh periode (butuh kolom tanggal),
   - rekurensi pada subsampel 80% transaksi (tanpa pengembalian).
7. **⑥ Rekomendasi & Unduh** — kalimat rekomendasi promosi dan unduhan CSV/Excel/JSON.

**Rumus**

- Support(X→Y) = n(X∪Y) / N
- Confidence(X→Y) = n(X∪Y) / n(X)
- Lift(X→Y) = Confidence(X→Y) / Support(Y)
- ΔI = |I_baru − I_lama| / I_lama × 100% (selisih absolut bila I_lama = 0)

**Catatan penafsiran**

- Apriori dan FP-Growth adalah algoritma eksak: pada data dan ambang yang sama keluarannya harus identik. Jika berbeda, periksa implementasi atau pengaturan.
- *Lift* sangat tinggi dari aturan yang hanya didukung beberapa transaksi belum tentu pola umum. Data tanpa ID pelanggan tidak bisa membedakan pembelian berulang oleh satu pelanggan.
- Memori Apriori mode default mlxtend bisa >1 GB pada ambang sangat rendah. Gunakan opsi **low_memory** di Streamlit Cloud.
