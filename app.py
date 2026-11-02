"""
Analisis Sensitivitas Apriori vs FP-Growth — aplikasi Streamlit
Penelitian Pemula LPPM Universitas Ngudi Waluyo 2026

Jalankan lokal:  streamlit run app.py
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src import data as D
from src import mining as M
from src import recommend as R
from src import robust as RB
from src import viz as V

st.set_page_config(page_title="Apriori vs FP-Growth — Analisis Sensitivitas", page_icon="🛒", layout="wide")

SAMPLE = Path(__file__).parent / "data" / "contoh_transaksi.csv"
DEFAULT_SUP = {"SKU": [0.1, 0.2, 0.3, 0.5, 1, 2, 3], "Departemen": [1, 2, 3, 5, 10], "Deskripsi": [0.1, 0.2, 0.3, 0.5, 1, 2, 3]}
DEFAULT_CONF = [10, 30, 50, 70, 90]


# ------------------------------------------------------------------ cache helpers
@st.cache_data(show_spinner=False)
def load_raw(raw: bytes) -> pd.DataFrame:
    return D.read_csv_auto(raw)


@st.cache_data(show_spinner="Membersihkan data…")
def clean(raw: bytes, mode: str, cols: tuple):
    df = D.read_csv_auto(raw)
    if mode == "pos":
        return D.clean_pos_log(df)
    return D.clean_generic(df, *cols)


@st.cache_data(show_spinner="Membentuk matriks transaksi–item…")
def matrix(items: pd.DataFrame, level: str, prefix: int):
    it = D.apply_granularity(items, level, prefix)
    return it, M.to_matrix(it)


def parse_pct_list(text: str) -> list[float]:
    vals = []
    for t in text.replace(";", ",").split(","):
        t = t.strip().replace("%", "").replace(" ", "")
        if t:
            vals.append(float(t.replace(",", ".")) / 100)
    return sorted(set(vals))


def fmt_pct(x, d=1):
    return f"{x*100:.{d}f}%".replace(".", ",")


# ------------------------------------------------------------------ sidebar
st.sidebar.title("🛒 Apriori vs FP-Growth")
st.sidebar.caption("Analisis sensitivitas parameter × granularitas item, dengan seleksi aturan robust.")

st.sidebar.header("1 · Data")
up = st.sidebar.file_uploader("Unggah CSV transaksi", type=["csv", "txt"],
                              help="Log POS toko (delimiter ';' dengan kolom FLAG, CODE, NOTA, TGL) atau CSV umum.")
use_sample = st.sidebar.checkbox("Pakai data contoh (sintetis)", value=up is None, disabled=up is not None)

raw = up.getvalue() if up is not None else (SAMPLE.read_bytes() if use_sample and SAMPLE.exists() else None)
if raw is None:
    st.title("Analisis Sensitivitas Apriori dan FP-Growth")
    st.info("Unggah file CSV transaksi di panel kiri, atau centang **Pakai data contoh** untuk mencoba.")
    st.stop()

df_raw = load_raw(raw)
pos = D.is_pos_log(df_raw)
if pos:
    st.sidebar.success("Format terdeteksi: **log POS toko**")
    mode, cols = "pos", ()
else:
    st.sidebar.info("Format umum — pilih kolom:")
    c = list(df_raw.columns)
    guess = lambda keys, default: next((i for i, x in enumerate(c) if any(k in x.lower() for k in keys)), default)
    trx_col = st.sidebar.selectbox("Kolom ID transaksi", c, index=guess(["nota", "trans", "invoice", "order"], 0))
    item_col = st.sidebar.selectbox("Kolom item (kode/nama)", c, index=guess(["kode", "code", "item", "sku"], min(1, len(c) - 1)))
    desc_opt = ["(tidak ada)"] + c
    desc_col = st.sidebar.selectbox("Kolom deskripsi (opsional)", desc_opt, index=guess(["nama", "desc", "produk"], -1) + 1)
    date_col = st.sidebar.selectbox("Kolom tanggal (opsional)", desc_opt, index=guess(["tgl", "tanggal", "date", "time"], -1) + 1)
    mode = "generic"
    cols = (trx_col, item_col, None if desc_col == "(tidak ada)" else desc_col, None if date_col == "(tidak ada)" else date_col)

cr = clean(raw, mode, cols)
items = cr.items

st.sidebar.header("2 · Granularitas item")
level = st.sidebar.radio("Tingkat item", ["SKU", "Departemen", "Deskripsi"], horizontal=True,
                         help="Departemen = n digit awal kode item (pada data toko: 3 digit).")
prefix = st.sidebar.number_input("Panjang prefiks departemen", 1, 10, 3, disabled=level != "Departemen")
it_level, X = matrix(items, level, int(prefix))

# nama item untuk tampilan
sku_names = D.item_names(items)
if level == "Departemen":
    if "dept_names" not in st.session_state or st.session_state.get("dept_prefix") != prefix:
        st.session_state.dept_names = D.default_dept_names(items, int(prefix))
        st.session_state.dept_prefix = prefix
    dn = st.session_state.dept_names
    names = {r.kode: (r.nama if isinstance(r.nama, str) and r.nama.strip() else f"Dept {r.kode}") for r in dn.itertuples()}
elif level == "Deskripsi":
    names = {}
else:
    names = sku_names

st.sidebar.header("3 · Info")
st.sidebar.metric("Transaksi", f"{X.shape[0]:,}".replace(",", "."))
st.sidebar.metric(f"Item ({level})", f"{X.shape[1]:,}".replace(",", "."))
st.sidebar.caption("Kode: github.com/[PERLU DIISI] · Lisensi MIT")

# ------------------------------------------------------------------ main
st.title("Analisis Sensitivitas Apriori dan FP-Growth")
st.caption("Minimum support × minimum confidence × granularitas item · seleksi aturan dengan uji signifikansi dan stabilitas")

tabs = st.tabs(["① Data & Pembersihan", "② Profil Data", "③ Sensitivitas Parameter", "④ Biaya Komputasi",
                "⑤ Aturan & Uji Robust", "⑥ Rekomendasi & Unduh", "ℹ️ Panduan"])

# ① ----------------------------------------------------------------
with tabs[0]:
    st.subheader("Tahapan pembersihan")
    st.dataframe(pd.DataFrame(cr.steps), hide_index=True, width="stretch")
    ch = cr.checks
    c1, c2, c3 = st.columns(3)
    if "ttl_sama_dengan_plu" in ch and ch["ttl_sama_dengan_plu"] is not None:
        c1.metric("Nota dengan TTL = Σ(QTY×PRICE) PLU", fmt_pct(ch["ttl_sama_dengan_plu"]),
                  help="Jika ~100%, baris PLU sudah bersih dari item batal (VD1) sehingga VD1 tidak perlu dikurangkan.")
    if "kode_multi_deskripsi" in ch:
        c2.metric("Kode dengan >1 deskripsi", ch["kode_multi_deskripsi"])
        c3.metric("Deskripsi dengan >1 kode", ch["deskripsi_multi_kode"])
    if ch.get("nota_tanpa_ttl"):
        st.warning(f"{len(ch['nota_tanpa_ttl'])} nota memiliki baris PLU tanpa TTL dan dibuang: {', '.join(ch['nota_tanpa_ttl'][:10])}")
    if "flag_counts" in ch:
        with st.expander("Jumlah baris per FLAG"):
            st.json(ch["flag_counts"])
    with st.expander("Contoh data mentah (20 baris)"):
        st.dataframe(df_raw.head(20), width="stretch")
    with st.expander("Contoh data bersih (20 baris)"):
        st.dataframe(items.head(20), width="stretch")
    st.download_button("⬇️ Unduh data bersih (CSV)", items.to_csv(index=False).encode(), "data_bersih.csv", "text/csv")

# ② ----------------------------------------------------------------
with tabs[1]:
    pf = M.profile(X)
    st.subheader(f"Profil data — granularitas {level}")
    k = st.columns(5)
    k[0].metric("Transaksi", f"{pf['transaksi']:,}".replace(",", "."))
    k[1].metric("Item", f"{pf['item']:,}".replace(",", "."))
    k[2].metric("Kepadatan matriks", fmt_pct(pf["kepadatan"], 2))
    k[3].metric("Rata-rata item/keranjang", f"{pf['rata_item']:.2f}".replace(".", ","))
    k[4].metric("Keranjang 1 item", fmt_pct(pf["satu_item"]))
    k = st.columns(4)
    k[0].metric("Support item tertinggi", fmt_pct(pf["support_maks"], 2))
    k[1].metric("Item support ≥ 1%", pf["item_ge_1"])
    k[2].metric("Item support ≥ 0,5%", pf["item_ge_05"])
    k[3].metric("Item support ≥ 0,1%", pf["item_ge_01"])

    sup_max = pf["support_maks"]
    if sup_max < 0.10:
        st.info(f"💡 Support item tertinggi hanya {fmt_pct(sup_max, 2)}. Ambang *minimum support* harus di bawah "
                f"nilai ini agar ada itemset berpasangan; untuk data seperti ini mulai grid dari ambang yang setara "
                f"±4–10 transaksi (≈ {fmt_pct(4/pf['transaksi'], 2)}–{fmt_pct(10/pf['transaksi'], 2)}).")
    a, b = st.columns(2)
    with a:
        st.markdown("**Item paling sering muncul**")
        st.plotly_chart(V.support_distribution(pf["support_series"], names), width="stretch")
    with b:
        st.markdown("**Berapa item yang lolos tiap ambang support?**")
        st.plotly_chart(V.support_ccdf(pf["support_series"]), width="stretch")
        st.markdown("**Ukuran keranjang**")
        st.plotly_chart(V.basket_size_hist(X), width="stretch")

    if level == "Departemen":
        st.subheader("Nama departemen")
        st.caption("Isi kolom **nama**. Label kosong ditampilkan sebagai 'Dept <kode>'. Produk dominan membantu menebak isi departemen.")
        st.session_state.dept_names = st.data_editor(st.session_state.dept_names, hide_index=True, width="stretch",
                                                     disabled=["kode", "produk_dominan", "transaksi"], key="dept_editor")
        up_names = st.file_uploader("…atau unggah CSV nama departemen (kolom: kode,nama)", type="csv", key="dn_up")
        if up_names is not None:
            m = pd.read_csv(up_names, dtype=str)
            dn = st.session_state.dept_names.drop(columns="nama").merge(m[["kode", "nama"]], on="kode", how="left")
            st.session_state.dept_names = dn[["kode", "nama", "produk_dominan", "transaksi"]]
            st.rerun()

# ③ ----------------------------------------------------------------
with tabs[2]:
    st.subheader("Eksperimen grid: minimum support × minimum confidence")
    n = X.shape[0]
    c1, c2 = st.columns(2)
    sup_txt = c1.text_input("Minimum support (%) — pisahkan dengan koma",
                            ", ".join(f"{v:g}" for v in DEFAULT_SUP[level]), key=f"sup_{level}")
    conf_txt = c2.text_input("Minimum confidence (%)", ", ".join(str(v) for v in DEFAULT_CONF), key="conf")
    sups, confs = parse_pct_list(sup_txt), parse_pct_list(conf_txt)
    st.caption("Setara jumlah transaksi minimum: " +
               " · ".join(f"{fmt_pct(s, 2)} → {M.min_count(s, n)} trx" for s in sups))

    with st.expander("⚙️ Pengaturan lanjutan", expanded=False):
        a1, a2, a3 = st.columns(3)
        reps = a1.number_input("Ulangan pengukuran waktu", 1, 20, 5, help="Waktu dilaporkan sebagai median.")
        meas_mem = a2.checkbox("Ukur memori puncak (tracemalloc)", True, help="Memperlambat eksekusi ±2×.")
        low_mem = a3.checkbox("Apriori mode low_memory", False,
                              help="Mode hemat memori mlxtend. Mode default menyimpan matriks kandidat penuh sehingga "
                                   "memorinya bisa >1 GB pada ambang sangat rendah.")
        b1, b2 = st.columns(2)
        floor = b1.number_input("Lewati Apriori untuk support < (%)", 0.0, 100.0, 0.0, 0.05,
                                help="Pengaman memori di server kecil (Streamlit Cloud ±1 GB). 0 = tidak dilewati.") / 100
        max_len = b2.number_input("Panjang itemset maksimum (0 = tanpa batas)", 0, 10, 0)
        if not low_mem and floor == 0 and level != "Departemen" and min(sups or [1]) < 0.003:
            st.warning("Apriori mode default pada support < 0,3% bisa memakan >1 GB memori. Di Streamlit Cloud aktifkan "
                       "**low_memory** atau isi batas *lewati Apriori*.")

    if st.button("▶️ Jalankan eksperimen", type="primary", disabled=not (sups and confs)):
        bar = st.progress(0.0, "Menyiapkan…")
        res, rulesets, eq = M.run_grid(X, sups, confs, int(reps), meas_mem, low_mem, int(max_len) or None,
                                       floor or None, progress=lambda p, t: bar.progress(p, t))
        bar.empty()
        st.session_state.grid = dict(res=res, eq=eq, level=level, n=n, low_mem=low_mem)

    g = st.session_state.get("grid")
    if g and g["level"] == level:
        res, eq = g["res"], g["eq"]
        if len(eq):
            ok = bool(eq["itemset_identik"].all() and eq["aturan_identik"].all())
            (st.success if ok else st.error)(
                f"Kesamaan keluaran Apriori vs FP-Growth: itemset & aturan identik di {int(eq['aturan_identik'].sum())}"
                f"/{len(eq)} ambang support." + ("" if ok else " ⚠️ Ada perbedaan — periksa implementasi/pengaturan."))
        algo_view = "FP-Growth"
        metric = st.selectbox("Nilai pada heatmap", ["aturan", "aturan_lift_gt1", "itemset", "itemset_k2", "rerata_lift",
                                                       "rerata_confidence"], index=0)
        st.plotly_chart(V.rules_heatmap(res, metric, algo_view), width="stretch")
        empty = res[(res.algoritma == algo_view) & (res.aturan == 0)]
        st.caption(f"{len(empty)} dari {len(res[res.algoritma == algo_view])} konfigurasi tidak menghasilkan aturan.")

        neg = res[(res.algoritma == algo_view) & (res.aturan > res.aturan_lift_gt1)]
        if len(neg):
            st.warning("Konfigurasi yang meloloskan aturan berasosiasi negatif (lift ≤ 1): " +
                       "; ".join(f"support {fmt_pct(r.min_support, 2)}, conf {fmt_pct(r.min_confidence, 0)}: "
                                 f"{r.aturan - r.aturan_lift_gt1}/{r.aturan}" for r in neg.itertuples()))

        a, b = st.columns(2)
        with a:
            cf = st.selectbox("Kurva respons terhadap support, pada confidence", sorted(res.min_confidence.unique()),
                              index=min(1, res.min_confidence.nunique() - 1), format_func=lambda v: fmt_pct(v, 0))
            yv = st.selectbox("Indikator", ["aturan", "itemset", "itemset_k2", "rerata_lift"], key="y1")
            st.plotly_chart(V.response_curve(res[res.algoritma == algo_view], "min_support", yv, "min_confidence",
                                             {"min_confidence": cf}), width="stretch")
        with b:
            sf = st.selectbox("Kurva respons terhadap confidence, pada support", sorted(res.min_support.unique()),
                              format_func=lambda v: fmt_pct(v, 2))
            yv2 = st.selectbox("Indikator", ["aturan", "aturan_lift_gt1", "rerata_lift", "rerata_confidence"], key="y2")
            st.plotly_chart(V.response_curve(res[res.algoritma == algo_view], "min_confidence", yv2, "min_support",
                                             {"min_support": sf}), width="stretch")

        st.markdown("**Perubahan indikator antar-ambang support** (ΔI = |I_baru − I_lama| / I_lama × 100%; selisih absolut bila I_lama = 0)")
        ind = st.selectbox("Indikator ΔI", ["aturan", "itemset", "itemset_k2"], key="dI")
        dt = M.delta_table(res, ind)
        st.dataframe(dt.style.format({"dari": "{:.2%}", "ke": "{:.2%}", "min_confidence": "{:.0%}", "perubahan": "{:.1f}"}),
                     hide_index=True, width="stretch")
        with st.expander("Tabel hasil lengkap"):
            st.dataframe(res, hide_index=True, width="stretch")
        st.download_button("⬇️ Unduh hasil grid (CSV)", res.to_csv(index=False).encode(), f"grid_{level}.csv", "text/csv")
    else:
        st.info("Atur parameter lalu tekan **Jalankan eksperimen**.")

# ④ ----------------------------------------------------------------
with tabs[3]:
    st.subheader("Biaya komputasi Apriori vs FP-Growth")
    g = st.session_state.get("grid")
    if g and g["level"] == level:
        res = g["res"]
        cf = st.selectbox("Pada minimum confidence", sorted(res.min_confidence.unique()),
                          index=min(1, res.min_confidence.nunique() - 1), format_func=lambda v: fmt_pct(v, 0), key="cfc")
        f1, f2 = V.compute_curves(res, cf)
        a, b = st.columns(2); a.plotly_chart(f1, width="stretch"); b.plotly_chart(f2, width="stretch")
        d = res[res.min_confidence == cf].pivot_table(index="min_support", columns="algoritma", values=["waktu_ms", "memori_mb"])
        st.dataframe(d.style.format("{:.2f}"), width="stretch")
        if {"Apriori", "FP-Growth"} <= set(res.algoritma):
            w = res[res.min_confidence == cf].pivot_table(index="min_support", columns="algoritma", values="waktu_ms").dropna()
            faster = np.where(w["Apriori"] < w["FP-Growth"], "Apriori", "FP-Growth")
            changes = [(w.index[i - 1], w.index[i]) for i in range(1, len(w)) if faster[i] != faster[i - 1]]
            if changes:
                lo, hi = changes[0]
                st.success(f"Titik silang waktu di antara support {fmt_pct(lo, 2)} dan {fmt_pct(hi, 2)}: "
                           f"di bawahnya {faster[0]} lebih cepat, di atasnya {faster[-1]} lebih cepat.")
            else:
                st.info(f"Tidak ada titik silang pada rentang ini — {faster[0]} lebih cepat di semua ambang.")
        st.caption(f"Mode Apriori: {'low_memory' if g['low_mem'] else 'default mlxtend'}. Angka absolut bergantung pada "
                   "mesin dan pustaka; laporkan spesifikasi mesin bersama hasil.")
    else:
        st.info("Jalankan eksperimen di tab ③ terlebih dahulu.")

# ⑤ ----------------------------------------------------------------
with tabs[4]:
    st.subheader("Aturan asosiasi dan seleksi aturan robust")
    st.caption("Robust = signifikan (Fisher satu sisi + koreksi Benjamini–Hochberg) **dan** muncul di dua paruh periode "
               "**dan** muncul di ≥ ambang rekurensi subsampel 80%.")
    d0 = {"SKU": (0.1, 30), "Departemen": (1.0, 30), "Deskripsi": (0.1, 30)}[level]
    c = st.columns(4)
    rs = c[0].number_input("Min support (%)", 0.01, 100.0, d0[0], 0.05, key=f"rs_{level}") / 100
    rc = c[1].number_input("Min confidence (%)", 1.0, 100.0, float(d0[1]), 5.0, key="rc") / 100
    B = c[2].number_input("Jumlah subsampel", 10, 500, 100, 10)
    frac = c[3].slider("Proporsi subsampel", 0.5, 0.95, 0.8, 0.05)
    c = st.columns(4)
    alpha = c[0].number_input("α (q BH)", 0.001, 0.2, 0.05, 0.005, format="%.3f")
    min_rec = c[1].slider("Rekurensi minimum", 0.1, 1.0, 0.5, 0.05)
    seed = c[2].number_input("Seed", 0, 99999, 42)
    use_t = c[3].checkbox("Uji dua paruh periode", items["date"].notna().any(), disabled=not items["date"].notna().any())
    split_how = st.radio("Cara membagi dua paruh", ["rentang", "median"], horizontal=True,
                         format_func=lambda v: {"rentang": "Tengah rentang tanggal (seperti artikel)",
                                                "median": "Median waktu transaksi"}[v], disabled=not use_t)
    st.caption(f"Ambang {fmt_pct(rs, 2)} setara {M.min_count(rs, X.shape[0])} transaksi.")

    if st.button("▶️ Tambang & uji aturan", type="primary"):
        bar = st.progress(0.0, "Menambang…")
        rr, info = RB.robust_rules(it_level, rs, rc, int(B), frac, int(seed), alpha, min_rec, None, use_t,
                                   progress=lambda p, t: bar.progress(p, t), split_how=split_how)
        bar.empty()
        st.session_state.robust = dict(r=rr, info=info, level=level, s=rs, c=rc)

    rb = st.session_state.get("robust")
    if rb and rb["level"] == level:
        info, rr = rb["info"], rb["r"]
        if not len(rr):
            st.warning("Tidak ada aturan pada konfigurasi ini. Turunkan minimum support (lihat tab ②) atau naikkan granularitas.")
        else:
            k = st.columns(5)
            k[0].metric("Aturan", info["n_rules"]); k[1].metric("Signifikan", info["signifikan"])
            k[2].metric("Muncul di dua paruh", info["dua_paruh"]); k[3].metric(f"Rekurensi ≥ {min_rec:.0%}", info["rekurensi_ok"])
            k[4].metric("Robust", info["robust"])
            if "batas_paruh" in info:
                st.caption(f"Paruh 1: {info['n_paruh1']} transaksi sebelum {info['batas_paruh']}; paruh 2: {info['n_paruh2']} transaksi.")
            if "catatan" in info:
                st.caption(info["catatan"])
            lr = R.label_rules(rr, names, show_code=st.checkbox("Tampilkan kode item", False))
            only = st.checkbox("Tampilkan hanya aturan robust", True)
            view = lr[lr.robust] if only else lr
            tbl = R.tidy_table(view)
            st.dataframe(tbl.style.format({"Support": "{:.3%}", "Confidence": "{:.0%}", "Lift": "{:.2f}",
                                           "q (BH)": "{:.4f}", "Rekurensi": "{:.0%}"}),
                         hide_index=True, width="stretch")
            st.plotly_chart(V.rule_scatter(lr), width="stretch")
            small = lr[lr.robust & (lr.n_bersama < 10)]
            if len(small):
                st.warning(f"{len(small)} aturan robust bertumpu pada < 10 transaksi. Tanpa ID pelanggan, pola ini bisa "
                           "berasal dari segelintir pelanggan tetap — konfirmasi ke pemilik toko.")
    else:
        st.info("Atur parameter lalu tekan **Tambang & uji aturan**.")

# ⑥ ----------------------------------------------------------------
with tabs[5]:
    st.subheader("Rekomendasi promosi")
    rb = st.session_state.get("robust")
    if rb and rb["level"] == level and len(rb["r"]):
        lr = R.label_rules(rb["r"], names)
        recs = R.recommendation_text(lr, level)
        if recs:
            for t in recs:
                st.markdown("- " + t)
        else:
            st.info("Belum ada aturan robust. Longgarkan ambang rekurensi atau ubah parameter di tab ⑤.")
        tbl = R.tidy_table(lr)
        st.download_button("⬇️ Unduh semua aturan + hasil uji (CSV)", tbl.to_csv(index=False).encode(),
                           f"aturan_{level}.csv", "text/csv")
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as xw:
            tbl.to_excel(xw, sheet_name="Aturan", index=False)
            pd.DataFrame(cr.steps).to_excel(xw, sheet_name="Pembersihan", index=False)
            g = st.session_state.get("grid")
            if g and g["level"] == level:
                g["res"].to_excel(xw, sheet_name="Grid", index=False)
        st.download_button("⬇️ Unduh laporan Excel", buf.getvalue(), f"laporan_{level}.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        summary = dict(level=level, min_support=rb["s"], min_confidence=rb["c"], **{k: v for k, v in rb["info"].items()})
        st.download_button("⬇️ Unduh ringkasan (JSON)", json.dumps(summary, indent=2, default=str).encode(),
                           f"ringkasan_{level}.json", "application/json")
    else:
        st.info("Jalankan **Tambang & uji aturan** di tab ⑤ terlebih dahulu.")

# ℹ️ ---------------------------------------------------------------
with tabs[6]:
    st.markdown(Path(__file__).with_name("PANDUAN.md").read_text(encoding="utf-8"))
