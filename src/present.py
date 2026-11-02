"""
Mode "Presentasi Toko": halaman-halaman ringkas berbahasa awam untuk pemilik toko.
Semua angka, grafik, dan kalimat dihitung dari data yang sedang dimuat.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from . import business as BZ
from . import data as D
from . import robust as RB

FONT = dict(size=16)
BLUE, GREY, ORANGE = "#1f6fb2", "#b8c2cc", "#e8833a"

CSS = """
<style>
.big-num {font-size: 2.6rem; font-weight: 700; line-height: 1.1; color: #1f3b57;}
.big-lbl {font-size: 1.0rem; color: #5b6b7a; margin-bottom: .2rem;}
.box {background:#f4f7fa; border-radius: 12px; padding: 18px 20px; height: 100%;}
.insight {font-size: 1.25rem; line-height: 1.55; margin: .35rem 0;}
.slide-title {font-size: 2.1rem; font-weight: 700; color:#1f3b57; margin-bottom: .2rem;}
.slide-sub {font-size: 1.1rem; color:#5b6b7a; margin-bottom: 1.2rem;}
.action {font-size: 1.2rem; line-height:1.5; background:#fff7ef; border-left: 6px solid #e8833a;
         padding: 12px 16px; border-radius: 8px; margin: .5rem 0;}
.note {font-size: .95rem; color:#6b7785;}
</style>
"""


def _kpi(col, label, value, sub=""):
    col.markdown(f'<div class="box"><div class="big-lbl">{label}</div><div class="big-num">{value}</div>'
                 f'<div class="note">{sub}</div></div>', unsafe_allow_html=True)


def _title(t, sub=""):
    st.markdown(f'<div class="slide-title">{t}</div><div class="slide-sub">{sub}</div>', unsafe_allow_html=True)


def _insight(t):
    st.markdown(f'<div class="insight">{t}</div>', unsafe_allow_html=True)


def _action(t):
    st.markdown(f'<div class="action">{t}</div>', unsafe_allow_html=True)


def _fig(fig, h=420):
    fig.update_layout(font=FONT, height=h, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="white")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def _md_bold(s: str) -> str:
    """**x** -> <b>x</b> agar bisa dipakai di dalam blok HTML."""
    import re
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)


# ----------------------------------------------------------------- aturan asosiasi (di-cache)
@st.cache_data(show_spinner="Mencari barang yang sering dibeli bersamaan…")
def _pairs(items: pd.DataFrame, level: str, s: float, c: float, prefix: int):
    it = D.apply_granularity(items, level, prefix)
    r, info = RB.robust_rules(it, s, c, B=50, frac=0.8, seed=42, alpha=0.05, min_recur=0.5, max_len=3,
                              use_temporal=items["date"].notna().any())
    return r, info


def _rule_rows(r: pd.DataFrame, names: dict, only_robust=True, k=6):
    if r is None or not len(r):
        return []
    d = r[r["robust"]] if only_robust and "robust" in r else r
    d = d[d["antecedents"].map(len) == 1]  # kalimat paling mudah dipahami: satu barang -> satu barang
    d = d.sort_values(["lift", "n_bersama"], ascending=False)
    seen, out = set(), []
    for _, x in d.iterrows():
        a = next(iter(x["antecedents"])); b = next(iter(x["consequents"]))
        key = frozenset([a, b])
        if key in seen:  # pasangan bolak-balik cukup ditampilkan sekali
            continue
        seen.add(key)
        out.append(dict(a=BZ.nice(names.get(a, a)), b=BZ.nice(names.get(b, b)), conf=x["confidence"], lift=x["lift"],
                        n=int(x["n_bersama"])))
        if len(out) >= k:
            break
    return out


# ----------------------------------------------------------------- render
def render(cr, names_sku: dict, names_dept: dict, prefix: int = 3):
    st.markdown(CSS, unsafe_allow_html=True)
    if cr.lines is None or cr.lines["price"].isna().all():
        st.warning("Mode presentasi membutuhkan kolom harga (dan sebaiknya jumlah & harga pokok). "
                   "Pilih kolom tersebut di panel kiri.")
        st.stop()

    full = BZ.prepare(cr.lines, prefix, names_dept)
    # ---- filter periode
    dmin, dmax = full["tanggal"].min(), full["tanggal"].max()
    if pd.notna(dmin) and dmin != dmax:
        rng = st.sidebar.date_input("Periode yang dipresentasikan", (dmin.date(), dmax.date()),
                                    min_value=dmin.date(), max_value=dmax.date())
        if isinstance(rng, (list, tuple)) and len(rng) == 2:
            full = full[(full["tanggal"] >= pd.Timestamp(rng[0])) & (full["tanggal"] <= pd.Timestamp(rng[1]))]
    d = full
    keep = set(d["trx"])
    items = cr.items[cr.items["trx"].isin(keep)]
    vd = cr.voids
    if vd is not None and len(vd):
        vdt = pd.to_datetime(vd["date"]).dt.normalize()
        vd = vd[(vdt >= d["tanggal"].min()) & (vdt <= d["tanggal"].max())]

    K = BZ.kpis(d)
    prod = BZ.products(d)
    prod["nama"] = prod["nama"].map(BZ.nice)
    dept = BZ.departments(d)
    bk = BZ.baskets(d)
    bk["solo"]["nama"] = bk["solo"]["nama"].map(BZ.nice)
    V = BZ.voids(vd, K["omzet"])
    cost_ok = "laba" in K
    time_ok = BZ.has_time(d)
    pk = BZ.peak_hours(d) if time_ok else None
    wd = BZ.by_weekday(d)
    par = BZ.pareto(prod)
    mix = BZ.profit_mix(dept) if cost_ok else None

    N = K["transaksi"]
    s_sku = max(0.001, 4 / N)
    r_dept, i_dept = _pairs(items, "Departemen", 0.01, 0.30, prefix)
    r_sku, i_sku = _pairs(items, "SKU", s_sku, 0.30, prefix)
    pairs_dept = _rule_rows(r_dept, names_dept, k=5)
    pairs_sku = _rule_rows(r_sku, names_sku)

    periode = f"{K['periode_awal']:%d %b %Y} – {K['periode_akhir']:%d %b %Y}"

    slides = ["Ringkasan", "Kapan toko ramai", "Produk andalan", "Kategori barang",
              "Laris tapi untung tipis", "Isi keranjang belanja", "Barang yang dibeli bersamaan",
              "Pembatalan transaksi", "Rencana aksi", "Catatan data"]
    if not cost_ok:
        slides.remove("Laris tapi untung tipis")
    if V is None:
        slides.remove("Pembatalan transaksi")

    if "slide" not in st.session_state:
        st.session_state.slide = 0
    st.session_state.slide = min(st.session_state.slide, len(slides) - 1)
    nav = st.columns([1, 6, 1])
    if nav[0].button("◀ Sebelumnya", width="stretch", disabled=st.session_state.slide == 0):
        st.session_state.slide -= 1; st.rerun()
    if nav[2].button("Berikutnya ▶", width="stretch", disabled=st.session_state.slide == len(slides) - 1):
        st.session_state.slide += 1; st.rerun()
    pick = nav[1].selectbox("Halaman", range(len(slides)), index=st.session_state.slide,
                            format_func=lambda i: f"{i+1}/{len(slides)} · {slides[i]}", label_visibility="collapsed")
    if pick != st.session_state.slide:
        st.session_state.slide = pick; st.rerun()
    page = slides[st.session_state.slide]
    st.divider()

    # =========================================================== 1. Ringkasan
    if page == "Ringkasan":
        _title("Ringkasan penjualan", f"Toko · {periode} · {K['hari_buka']} hari")
        c = st.columns(4)
        _kpi(c[0], "Omzet", BZ.rp(K["omzet"], True), f"≈ {BZ.rp(K['omzet_per_hari'], True)} per hari")
        if cost_ok:
            _kpi(c[1], "Laba kotor (perkiraan)", BZ.rp(K["laba_est"], True), f"margin rata-rata {BZ.pct(K['margin'], 1)}")
        else:
            _kpi(c[1], "Barang terjual", BZ.num(K["unit"]), "unit")
        _kpi(c[2], "Jumlah struk", BZ.num(K["transaksi"]), f"≈ {BZ.num(K['trx_per_hari'])} per hari")
        _kpi(c[3], "Belanja per struk", BZ.rp(K["rata_struk"], True), f"separuh struk ≤ {BZ.rp(K['median_struk'], True)}")
        st.write("")
        st.markdown("#### Hal-hal terpenting")
        pts = []
        if mix:
            pts.append(f"**{' & '.join(mix['om']['dept_nama'])}** menghasilkan {BZ.pct(mix['om_share'])} omzet, "
                       f"tetapi hanya **{BZ.pct(mix['om_laba'])} laba**. Sebaliknya **{' & '.join(mix['lb']['dept_nama'])}** "
                       f"hanya {BZ.pct(mix['lb_share'])} omzet namun menyumbang **{BZ.pct(mix['lb_laba'])} laba**.")
        pts.append(f"**{BZ.pct(par['porsi_produk'])} produk** ({par['n_produk']} dari {par['total_produk']}) "
                   f"menghasilkan **80% omzet**. Ketersediaan stok produk-produk ini paling menentukan.")
        if time_ok:
            a, b = pk["blok"]
            pts.append(f"Toko paling ramai pukul **{a:02d}.00–{b:02d}.00**; jam-jam ini menyumbang "
                       f"**{BZ.pct(pk['porsi_omzet_blok'])} omzet**.")
        pts.append(f"**{BZ.pct(bk['satu_item'])} struk hanya berisi satu jenis barang** — peluang besar menawarkan "
                   f"barang pelengkap di kasir.")
        for p in pts:
            _insight("• " + _md_bold(p))

    # =========================================================== 2. Waktu
    elif page == "Kapan toko ramai":
        _title("Kapan toko ramai?", "Rata-rata jumlah struk per jam dan per hari")
        if time_ok:
            h = BZ.hour_heat(d)
            fig = go.Figure(go.Heatmap(z=h.values, x=[f"{j:02d}" for j in h.columns], y=h.index,
                                       colorscale="Blues", text=np.vectorize(lambda v: BZ.num(v, 1))(h.values), texttemplate="%{text}",
                                       hovertemplate="%{y} jam %{x}.00: %{z:.1f} struk<extra></extra>"))
            fig.update_yaxes(autorange="reversed"); fig.update_xaxes(title="Jam")
            _fig(fig, 380)
            a, b = pk["blok"]
            _insight(_md_bold(f"• Jam tersibuk: **pukul {pk['jam_tersibuk']:02d}.00** (rata-rata "
                              f"{BZ.num(pk['trx_jam_tersibuk'], 0)} struk per jam)."))
            _insight(_md_bold(f"• Blok ramai **{a:02d}.00–{b:02d}.00** menyumbang **{BZ.pct(pk['porsi_omzet_blok'])} omzet**."))
            _insight(_md_bold(f"• Paling sepi sekitar **pukul {pk['jam_sepi']:02d}.00** — waktu yang cocok untuk "
                              f"isi ulang rak dan merapikan stok."))
        top = wd.sort_values("omzet", ascending=False)
        fig = px.bar(wd, x="hari", y="omzet", text=wd["omzet"].map(lambda v: BZ.rp(v, True)),
                     color_discrete_sequence=[BLUE])
        fig.update_layout(yaxis_title="Rata-rata omzet per hari", xaxis_title="")
        _fig(fig, 320)
        lo = top.iloc[-1]; hi = top.iloc[0]
        _insight(_md_bold(f"• Hari paling ramai: **{hi['hari']}** ({BZ.rp(hi['omzet'], True)} per hari), "
                          f"paling sepi: **{lo['hari']}** ({BZ.rp(lo['omzet'], True)}). "
                          f"Selisihnya {BZ.pct(hi['omzet']/lo['omzet']-1)}."))
        st.caption(f"Setiap hari dalam seminggu muncul {wd['n_hari'].min()}–{wd['n_hari'].max()} kali dalam periode ini.")

    # =========================================================== 3. Produk
    elif page == "Produk andalan":
        _title("Produk andalan", "Produk yang paling banyak menghasilkan uang")
        c1, c2 = st.columns(2)
        t10 = prod.head(10).iloc[::-1]
        fig = px.bar(t10, x="omzet", y="nama", orientation="h", text=t10["omzet"].map(lambda v: BZ.rp(v, True)),
                     color_discrete_sequence=[BLUE])
        fig.update_layout(title="10 produk dengan omzet terbesar", xaxis_title="", yaxis_title="")
        with c1: _fig(fig, 460)
        if cost_ok:
            l10 = prod.dropna(subset=["laba"]).sort_values("laba", ascending=False).head(10).iloc[::-1]
            fig = px.bar(l10, x="laba", y="nama", orientation="h", text=l10["laba"].map(lambda v: BZ.rp(v, True)),
                         color_discrete_sequence=[ORANGE])
            fig.update_layout(title="10 produk dengan laba kotor terbesar", xaxis_title="", yaxis_title="")
            with c2: _fig(fig, 460)
        else:
            u10 = prod.sort_values("unit", ascending=False).head(10).iloc[::-1]
            fig = px.bar(u10, x="unit", y="nama", orientation="h", color_discrete_sequence=[ORANGE])
            fig.update_layout(title="10 produk paling banyak terjual (unit)", xaxis_title="", yaxis_title="")
            with c2: _fig(fig, 460)
        top1 = prod.iloc[0]
        _insight(_md_bold(f"• **{par['n_produk']} produk** ({BZ.pct(par['porsi_produk'])} dari {par['total_produk']} produk "
                          f"yang terjual) menghasilkan **80% omzet**. Jangan sampai produk-produk ini kosong."))
        _insight(_md_bold(f"• Lebih tajam lagi: **separuh omzet** datang dari hanya **{par['n50']} produk**."))
        _insight(_md_bold(f"• Produk nomor satu: **{top1['nama']}** — {BZ.rp(top1['omzet'], True)} "
                          f"({BZ.pct(top1['porsi_omzet'], 1)} omzet), dibeli di {BZ.num(top1['struk'])} struk."))
        with st.expander("Daftar lengkap produk"):
            cols = ["nama", "dept", "omzet", "unit", "struk"] + (["laba", "margin"] if cost_ok else [])
            st.dataframe(prod[cols].style.format({"omzet": BZ.rp, "laba": BZ.rp, "margin": "{:.1%}", "unit": "{:,.0f}"}),
                         hide_index=True, width="stretch")

    # =========================================================== 4. Kategori
    elif page == "Kategori barang":
        _title("Kategori barang", "Sumbangan tiap kategori terhadap omzet" + (" dan laba" if cost_ok else ""))
        top = dept.head(12)
        if cost_ok:
            fig = go.Figure()
            fig.add_bar(y=top["dept_nama"][::-1], x=top["omzet"][::-1], orientation="h", name="Omzet",
                        marker_color=BLUE, text=top["omzet"][::-1].map(lambda v: BZ.rp(v, True)), textposition="outside")
            fig.add_bar(y=top["dept_nama"][::-1], x=top["laba"][::-1], orientation="h", name="Laba kotor",
                        marker_color=ORANGE)
            fig.update_layout(barmode="overlay", legend=dict(orientation="h", y=1.08))
            fig.update_traces(cliponaxis=False)
            fig.update_xaxes(range=[0, top["omzet"].max() * 1.18])
        else:
            fig = px.bar(top[::-1], x="omzet", y="dept_nama", orientation="h", color_discrete_sequence=[BLUE])
        fig.update_layout(xaxis_title="", yaxis_title="")
        _fig(fig, 520)
        a = dept.iloc[0]
        _insight(_md_bold(f"• Kategori terbesar: **{a['dept_nama']}** — {BZ.pct(a['porsi_omzet'])} omzet, ada di "
                          f"{BZ.pct(a['porsi_struk'])} struk."))
        if cost_ok:
            dd = dept[dept["porsi_omzet"] >= 0.02].dropna(subset=["margin"])
            hi = dd.sort_values("margin", ascending=False).iloc[0]; lo = dd.sort_values("margin").iloc[0]
            _insight(_md_bold(f"• Margin tertinggi (kategori ≥ 2% omzet): **{hi['dept_nama']}** ({BZ.pct(hi['margin'], 1)}). "
                              f"Terendah: **{lo['dept_nama']}** ({BZ.pct(lo['margin'], 1)})."))
            if mix:
                _insight(_md_bold(f"• Omzet terbesar dari **{' & '.join(mix['om']['dept_nama'])}** ({BZ.pct(mix['om_share'])} omzet, "
                                  f"{BZ.pct(mix['om_laba'])} laba). Laba justru banyak datang dari **{' & '.join(mix['lb']['dept_nama'])}** "
                                  f"({BZ.pct(mix['lb_share'])} omzet, {BZ.pct(mix['lb_laba'])} laba)."))
            _insight(_md_bold("• Kategori bermargin tinggi layak mendapat tempat rak terbaik (setinggi mata, dekat kasir)."))
        st.caption("Nama kategori disusun dari produk dominan tiap kode barang; dapat diubah di panel kiri.")

    # =========================================================== 5. Margin tipis
    elif page == "Laris tapi untung tipis":
        _title("Laris, tapi untungnya tipis", "Produk 30 teratas menurut omzet yang marginnya di bawah median")
        thin = BZ.thin_margin_bestsellers(prod)
        p30 = prod.head(30).dropna(subset=["margin"])
        med = p30["margin"].median()
        show = thin.head(10).iloc[::-1]
        fig = go.Figure(go.Bar(y=show["nama"], x=show["omzet"], orientation="h", marker_color=ORANGE,
                               text=[f"margin {BZ.pct(m, 1)}" for m in show["margin"]], textposition="outside",
                               cliponaxis=False))
        fig.update_layout(title=f"Produk laris dengan margin di bawah {BZ.pct(med, 1)}", xaxis_title="Omzet", yaxis_title="")
        fig.update_xaxes(range=[0, show["omzet"].max() * 1.3])
        _fig(fig, 460)
        if len(thin):
            tot = thin["omzet"].sum(); lb = thin["laba"].sum()
            _insight(_md_bold(f"• **{len(thin)} produk laris** bermargin di bawah {BZ.pct(med, 1)}: mereka menghasilkan "
                              f"**{BZ.pct(tot/K['omzet'])} omzet** toko, tetapi hanya **{BZ.pct(lb/K['laba'])} laba**."))
            _insight(_md_bold(f"• Jika margin produk-produk ini naik 1 poin saja (mis. dari harga beli yang lebih murah), "
                              f"laba bertambah ≈ **{BZ.rp(tot*0.01, True)}** per periode."))
            _insight(_md_bold("• Produk seperti ini biasanya penarik pengunjung (rokok, gula, air mineral). Jangan dinaikkan "
                              "sembarangan — cek harga pemasok lain atau pasangkan dengan produk bermargin tinggi."))
        st.caption("Laba kotor = (harga jual − harga pokok) × jumlah, memakai kolom harga pokok pada data kasir.")

    # =========================================================== 6. Keranjang
    elif page == "Isi keranjang belanja":
        _title("Isi keranjang belanja", "Berapa banyak dan berapa nilai yang dibeli dalam satu struk")
        c = st.columns(3)
        _kpi(c[0], "Struk berisi 1 jenis barang", BZ.pct(bk["satu_item"]))
        _kpi(c[1], "Belanja rata-rata (1 barang)", BZ.rp(bk["rata_single"], True))
        _kpi(c[2], "Belanja rata-rata (≥ 2 barang)", BZ.rp(bk["rata_multi"], True))
        st.write("")
        c1, c2 = st.columns(2)
        dist = bk["dist"]
        fig = px.bar(dist, x="kelas", y="struk", text="struk", color_discrete_sequence=[BLUE])
        fig.update_layout(title="Jumlah struk menurut nilai belanja", xaxis_title="", yaxis_title="")
        with c1: _fig(fig, 360)
        solo = bk["solo"].head(8).iloc[::-1]
        fig = px.bar(solo, x="struk", y="nama", orientation="h", color_discrete_sequence=[ORANGE])
        fig.update_layout(title="Barang yang paling sering dibeli sendirian", xaxis_title="Jumlah struk", yaxis_title="")
        with c2: _fig(fig, 360)
        gap = bk["rata_multi"] - bk["rata_single"]
        _insight(_md_bold(f"• Hampir **separuh pembeli** hanya membeli satu jenis barang, dengan belanja rata-rata "
                          f"{BZ.rp(bk['rata_single'], True)}."))
        add = 0.10 * bk["satu_item"] * K["transaksi"] * 5000
        _insight(_md_bold(f"• Jika 1 dari 10 pembeli ini ditawari dan menambah barang senilai Rp 5.000, omzet bertambah "
                          f"≈ **{BZ.rp(add, True)}** per periode."))
        _insight(_md_bold(f"• Pembeli yang paling sering datang untuk satu barang saja: **{bk['solo'].iloc[0]['nama']}** — "
                          f"titik yang tepat untuk tawaran barang pelengkap."))

    # =========================================================== 7. Pasangan
    elif page == "Barang yang dibeli bersamaan":
        _title("Barang yang sering dibeli bersamaan", "Hanya pola yang lolos uji statistik dan muncul berulang")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Antar-kategori → penataan rak")
            if pairs_dept:
                for p in pairs_dept:
                    _insight("• " + _md_bold(BZ.rule_sentence(p["a"], p["b"], p["conf"], p["lift"], p["n"])))
            else:
                st.info("Belum ada pola kategori yang cukup kuat pada periode ini.")
        with c2:
            st.markdown("#### Antar-produk → paket promosi")
            if pairs_sku:
                for p in pairs_sku:
                    _insight("• " + _md_bold(BZ.rule_sentence(p["a"], p["b"], p["conf"], p["lift"], p["n"])))
            else:
                st.info("Belum ada pola produk yang cukup kuat pada periode ini.")
        st.caption("“× lebih sering daripada pembeli lain” membandingkan peluang membeli barang kedua pada pembeli barang "
                   "pertama dengan peluangnya pada semua pembeli (lift). Pola produk didukung sedikit struk; uji coba dulu "
                   "dalam skala kecil sebelum dijadikan program tetap.")

    # =========================================================== 8. Void
    elif page == "Pembatalan transaksi":
        _title("Pembatalan barang di kasir", "Barang yang sudah dipindai lalu dibatalkan")
        c = st.columns(3)
        _kpi(c[0], "Nilai dibatalkan", BZ.rp(V["total"], True), f"{BZ.pct(V['porsi'], 1)} dari omzet")
        _kpi(c[1], "Jumlah pembatalan", BZ.num(V["baris"]), f"di {BZ.num(V['nota'])} struk")
        _kpi(c[2], "Nilai tipikal", BZ.rp(V["median"], True), "median per pembatalan")
        st.write("")
        ph = V["per_hari"]
        fig = px.bar(ph, x="tanggal", y="nilai", color_discrete_sequence=[GREY])
        fig.update_layout(yaxis_title="Nilai dibatalkan", xaxis_title="")
        _fig(fig, 300)
        _insight(_md_bold(f"• Pembatalan terbesar: **{V['terbesar_desc']}** sebanyak {BZ.num(V['terbesar_qty'])} unit "
                          f"({BZ.rp(V['terbesar'], True)}) — kemungkinan salah ketik jumlah. Tanpa satu kejadian ini, "
                          f"nilai pembatalan {BZ.pct(V['tanpa_terbesar'], 1)} dari omzet."))
        _insight(_md_bold("• Sebagian besar pembatalan bernilai kecil dan wajar (salah pindai, pembeli batal). "
                          "Yang perlu dijaga adalah pembatalan bernilai besar."))
        _action("Saran: pembatalan di atas nilai tertentu (mis. Rp 50.000) dicatat alasannya atau perlu persetujuan "
                "pemilik/penanggung jawab shift.")

    # =========================================================== 9. Aksi
    elif page == "Rencana aksi":
        _title("Rencana aksi", "Langkah sederhana yang bisa dicoba bulan depan, dan cara mengukurnya")
        acts = []
        acts.append(f"<b>Jaga stok {par['n50']} produk teratas.</b> Separuh omzet toko berasal dari {par['n50']} produk ini. "
                    f"Buat daftar cek stok harian khusus untuk mereka. <i>Ukur:</i> berapa kali produk ini kosong per minggu.")
        if time_ok:
            a, b = pk["blok"]
            acts.append(f"<b>Siapkan toko sebelum pukul {a:02d}.00.</b> Rak terisi dan kasir siap di jam ramai "
                        f"{a:02d}.00–{b:02d}.00; isi ulang stok di jam sepi (sekitar {pk['jam_sepi']:02d}.00). "
                        f"<i>Ukur:</i> antrean dan omzet jam ramai.")
        if pairs_sku:
            p = pairs_sku[0]
            acts.append(f"<b>Uji paket {p['a']} + {p['b']}.</b> Pasangan ini sering dibeli bersama. Coba potongan kecil "
                        f"selama 2 minggu. <i>Ukur:</i> jumlah paket terjual dibanding sebelumnya.")
        if pairs_dept:
            p = pairs_dept[0]
            acts.append(f"<b>Dekatkan rak {p['a']} dan {p['b']}.</b> Kedua kategori sering ada di struk yang sama. "
                        f"<i>Ukur:</i> struk yang memuat keduanya.")
        acts.append(f"<b>Tawarkan barang pelengkap di kasir.</b> {BZ.pct(bk['satu_item'])} struk hanya berisi satu "
                    f"jenis barang. <i>Ukur:</i> rata-rata jumlah barang per struk.")
        if cost_ok:
            thin = BZ.thin_margin_bestsellers(prod)
            if len(thin):
                acts.append(f"<b>Tinjau harga beli {len(thin)} produk laris bermargin tipis</b>, mulai dari "
                            f"{thin.iloc[0]['nama']}. <i>Ukur:</i> margin rata-rata kategori.")
        if V is not None:
            acts.append("<b>Catat alasan pembatalan bernilai besar.</b> <i>Ukur:</i> nilai pembatalan per bulan.")
        for i, a in enumerate(acts, 1):
            _action(f"{i}. {a}")

    # =========================================================== 10. Catatan
    elif page == "Catatan data":
        _title("Catatan tentang data", "Agar angka dibaca dengan tepat")
        notes = [f"Periode data: <b>{periode}</b> ({K['hari_buka']} hari), {BZ.num(K['transaksi'])} struk, "
                 f"{BZ.num(K['sku'])} produk terjual.",
                 "Hanya transaksi yang selesai (punya total pembayaran) yang dihitung; struk yang dibatalkan seluruhnya tidak ikut."]
        if cost_ok:
            notes.append(f"Laba kotor dihitung dari kolom harga pokok, yang tersedia untuk {BZ.pct(K['cakupan_hpp'])} omzet; "
                         f"sisanya diperkirakan dengan margin rata-rata. Mohon konfirmasi bahwa kolom tersebut memang harga pokok.")
        notes.append("Kategori barang disusun dari tiga digit awal kode barang dan produk dominannya, bukan dari daftar "
                     "kategori resmi toko.")
        notes.append("Pola “dibeli bersamaan” hanya ditampilkan bila lolos uji statistik dan muncul di dua paruh periode; "
                     "pola produk didukung sedikit struk sehingga perlu diuji coba dulu.")
        notes.append("Data tidak memuat identitas pelanggan, jadi pembelian berulang oleh pelanggan yang sama tidak bisa dibedakan.")
        for n in notes:
            _insight("• " + n)
