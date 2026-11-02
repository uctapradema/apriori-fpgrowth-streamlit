"""
Analisis bisnis untuk presentasi ke pemilik toko.

Semua angka dan kalimat temuan dihitung dari data, sehingga otomatis menyesuaikan bila data diganti
(mis. data bulan/tahun terbaru).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HARI = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]


# ----------------------------------------------------------------- format angka Indonesia
def rp(x: float, short: bool = False) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    if short:
        if abs(x) >= 1e9: return f"Rp {x/1e9:,.1f} M".replace(",", "X").replace(".", ",").replace("X", ".")
        if abs(x) >= 1e6: return f"Rp {x/1e6:,.1f} jt".replace(",", "X").replace(".", ",").replace("X", ".")
        if abs(x) >= 1e3: return f"Rp {x/1e3:,.0f} rb".replace(",", ".")
    return f"Rp {x:,.0f}".replace(",", ".")


def num(x: float, d: int = 0) -> str:
    s = f"{x:,.{d}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def pct(x: float, d: int = 0) -> str:
    return f"{x*100:.{d}f}%".replace(".", ",")


# ----------------------------------------------------------------- persiapan
def prepare(lines: pd.DataFrame, dept_len: int = 3, dept_names: dict | None = None) -> pd.DataFrame:
    """Tambahkan kolom omzet, hpp, laba, jam, hari, departemen."""
    d = lines.copy()
    d["qty"] = d["qty"].fillna(1.0)
    d["omzet"] = d["qty"] * d["price"]
    d["hpp"] = d["qty"] * d["cost"]
    d["laba"] = d["omzet"] - d["hpp"]
    d["date"] = pd.to_datetime(d["date"])
    d["tanggal"] = d["date"].dt.normalize()
    d["jam"] = d["date"].dt.hour
    d["hari"] = d["date"].dt.dayofweek
    d["dept"] = d["item"].astype(str).str[:dept_len]
    names = dept_names or {}
    d["dept_nama"] = d["dept"].map(lambda k: names.get(k, f"Dept {k}"))
    return d


def has_time(d: pd.DataFrame) -> bool:
    """True bila stempel waktu memuat jam (bukan hanya tanggal)."""
    return d["date"].notna().any() and (d["date"].dt.hour > 0).mean() > 0.5


def has_cost(d: pd.DataFrame) -> bool:
    return d["cost"].notna().mean() > 0.5 and d["price"].notna().any()


# ----------------------------------------------------------------- ringkasan
def kpis(d: pd.DataFrame) -> dict:
    trx = d.groupby("trx").agg(omzet=("omzet", "sum"), n_item=("item", "nunique"), unit=("qty", "sum"))
    days = d["tanggal"].nunique()
    k = dict(
        periode_awal=d["tanggal"].min(), periode_akhir=d["tanggal"].max(), hari_buka=days,
        omzet=d["omzet"].sum(), transaksi=len(trx), unit=d["qty"].sum(), sku=d["item"].nunique(),
        rata_struk=trx["omzet"].mean(), median_struk=trx["omzet"].median(),
        item_per_struk=trx["n_item"].mean(), struk_1item=(trx["n_item"] == 1).mean(),
        omzet_per_hari=d["omzet"].sum() / max(days, 1), trx_per_hari=len(trx) / max(days, 1),
    )
    if has_cost(d):
        c = d[d["cost"].notna()]
        k.update(laba=c["laba"].sum(), margin=c["laba"].sum() / c["omzet"].sum(),
                 cakupan_hpp=c["omzet"].sum() / d["omzet"].sum(),
                 laba_est=d["omzet"].sum() * c["laba"].sum() / c["omzet"].sum())
    return k


def daily(d: pd.DataFrame) -> pd.DataFrame:
    g = d.groupby("tanggal").agg(omzet=("omzet", "sum"), transaksi=("trx", "nunique"))
    g["hari"] = g.index.dayofweek.map(lambda i: HARI[i])
    return g.reset_index()


def by_weekday(d: pd.DataFrame) -> pd.DataFrame:
    """Rata-rata per hari dalam seminggu (dibagi jumlah tanggal untuk hari tsb)."""
    day = daily(d)
    day["dow"] = day["tanggal"].dt.dayofweek
    g = day.groupby("dow").agg(omzet=("omzet", "mean"), transaksi=("transaksi", "mean"), n_hari=("tanggal", "count"))
    g["hari"] = [HARI[i] for i in g.index]
    return g.reset_index()


def hour_heat(d: pd.DataFrame) -> pd.DataFrame:
    """Rata-rata jumlah transaksi per jam per hari-dalam-seminggu."""
    t = d.drop_duplicates("trx")
    n_days = t.groupby("hari")["tanggal"].nunique()
    h = t.groupby(["hari", "jam"]).size().unstack(fill_value=0)
    h = h.div(n_days, axis=0)
    h.index = [HARI[i] for i in h.index]
    return h


def peak_hours(d: pd.DataFrame) -> dict:
    t = d.drop_duplicates("trx")
    per_jam = t.groupby("jam").size() / t["tanggal"].nunique()
    om_jam = d.groupby("jam")["omzet"].sum()
    top = per_jam.sort_values(ascending=False)
    # blok jam ramai: jam dengan transaksi >= 1,25x rata-rata jam buka
    busy = sorted(per_jam[per_jam >= 1.25 * per_jam.mean()].index)
    blocks, cur = [], []
    for h in busy:
        if cur and h != cur[-1] + 1:
            blocks.append(cur); cur = []
        cur.append(h)
    if cur: blocks.append(cur)
    main = max(blocks, key=lambda b: per_jam[b].sum()) if blocks else [int(top.index[0])]
    share = om_jam[main].sum() / om_jam.sum()
    return dict(per_jam=per_jam, jam_tersibuk=int(top.index[0]), trx_jam_tersibuk=float(top.iloc[0]),
                blok=(min(main), max(main) + 1), porsi_omzet_blok=share,
                jam_sepi=int(per_jam.idxmin()), buka=(int(per_jam.index.min()), int(per_jam.index.max()) + 1))


# ----------------------------------------------------------------- produk
def products(d: pd.DataFrame) -> pd.DataFrame:
    name = d.groupby("item")["desc"].agg(lambda s: s.value_counts().index[0])
    g = d.groupby("item").agg(omzet=("omzet", "sum"), unit=("qty", "sum"), struk=("trx", "nunique"),
                              dept=("dept_nama", "first"))
    if has_cost(d):
        c = d[d["cost"].notna()].groupby("item").agg(laba=("laba", "sum"), omzet_c=("omzet", "sum"))
        g = g.join(c)
        g["margin"] = g["laba"] / g["omzet_c"]
    g["nama"] = name
    g["porsi_omzet"] = g["omzet"] / g["omzet"].sum()
    return g.sort_values("omzet", ascending=False).reset_index()


def pareto(prod: pd.DataFrame, target: float = 0.8) -> dict:
    cum = prod["omzet"].cumsum() / prod["omzet"].sum()
    n = int((cum < target).sum()) + 1
    n50 = int((cum < 0.5).sum()) + 1
    return dict(n_produk=n, total_produk=len(prod), porsi_produk=n / len(prod), target=target, kurva=cum.values,
                n50=n50)


def departments(d: pd.DataFrame) -> pd.DataFrame:
    g = d.groupby(["dept", "dept_nama"]).agg(omzet=("omzet", "sum"), struk=("trx", "nunique"),
                                             sku=("item", "nunique")).reset_index()
    if has_cost(d):
        c = d[d["cost"].notna()].groupby("dept").agg(laba=("laba", "sum"), omzet_c=("omzet", "sum"))
        g = g.merge(c, on="dept", how="left"); g["margin"] = g["laba"] / g["omzet_c"]
    g["porsi_omzet"] = g["omzet"] / g["omzet"].sum()
    g["porsi_struk"] = g["struk"] / d["trx"].nunique()
    return g.sort_values("omzet", ascending=False).reset_index(drop=True)


def thin_margin_bestsellers(prod: pd.DataFrame, top_n: int = 30) -> pd.DataFrame:
    """Produk laris (top-N omzet) dengan margin di bawah median margin seluruh produk laris."""
    if "margin" not in prod:
        return pd.DataFrame()
    top = prod.head(top_n).dropna(subset=["margin"])
    thr = top["margin"].median()
    return top[top["margin"] < thr].sort_values("omzet", ascending=False)


# ----------------------------------------------------------------- keranjang
def baskets(d: pd.DataFrame) -> dict:
    t = d.groupby("trx").agg(omzet=("omzet", "sum"), n_item=("item", "nunique"))
    single = t[t["n_item"] == 1]; multi = t[t["n_item"] > 1]
    # barang yang paling sering dibeli sendirian
    solo = d[d["trx"].isin(single.index)].groupby("item").agg(struk=("trx", "nunique"),
                                                              nama=("desc", lambda s: s.value_counts().index[0]))
    solo = solo.sort_values("struk", ascending=False)
    bins = [0, 5000, 10000, 20000, 50000, 100000, np.inf]
    labels = ["< 5 rb", "5–10 rb", "10–20 rb", "20–50 rb", "50–100 rb", "≥ 100 rb"]
    t["kelas"] = pd.cut(t["omzet"], bins=bins, labels=labels, right=False)
    dist = t.groupby("kelas", observed=False).agg(struk=("omzet", "size"), omzet=("omzet", "sum")).reset_index()
    return dict(tabel=t, satu_item=len(single) / len(t), rata_single=single["omzet"].mean(),
                rata_multi=multi["omzet"].mean(), solo=solo, dist=dist,
                n_item_dist=t["n_item"].clip(upper=10).value_counts().sort_index())


# ----------------------------------------------------------------- pembatalan
def voids(v: pd.DataFrame | None, omzet_total: float) -> dict | None:
    if v is None or v.empty:
        return None
    v = v.copy(); v["nilai"] = v["qty"] * v["price"]; v["tanggal"] = pd.to_datetime(v["date"]).dt.normalize()
    top1 = v["nilai"].max()
    per_hari = v.groupby("tanggal").agg(nilai=("nilai", "sum"), baris=("nilai", "size")).reset_index()
    top_item = v.groupby("desc").agg(nilai=("nilai", "sum"), kali=("nilai", "size")).sort_values("nilai", ascending=False)
    return dict(total=v["nilai"].sum(), baris=len(v), nota=v["trx"].nunique(), porsi=v["nilai"].sum() / omzet_total,
                terbesar=top1, terbesar_desc=v.loc[v["nilai"].idxmax(), "desc"],
                terbesar_qty=v.loc[v["nilai"].idxmax(), "qty"],
                tanpa_terbesar=(v["nilai"].sum() - top1) / omzet_total, median=v["nilai"].median(),
                per_hari=per_hari, top_item=top_item.reset_index())


# ----------------------------------------------------------------- kalimat aturan untuk pemilik toko
def rule_sentence(antecedent: str, consequent: str, conf: float, lift: float, n: int) -> str:
    per10 = max(1, round(conf * 10))
    if lift >= 10:
        kuat = "jauh di atas kebiasaan pembeli lain"
    else:
        kuat = f"{num(lift, 1)}× lebih sering daripada pembeli lain"
    return (f"Dari setiap 10 pembeli **{antecedent}**, sekitar **{per10}** juga membeli **{consequent}** "
            f"({kuat}; terlihat di {n} struk).")


def nice(name: str) -> str:
    """Nama produk POS (huruf kapital semua) -> huruf awal kapital agar mudah dibaca."""
    return " ".join(w if any(ch.isdigit() for ch in w) else w.capitalize() for w in str(name).split())


def profit_mix(dept: pd.DataFrame) -> dict | None:
    """Bandingkan porsi omzet dan porsi laba kategori: penyumbang omzet vs penyumbang laba."""
    if "laba" not in dept:
        return None
    g = dept.dropna(subset=["laba"]).copy()
    g["porsi_laba"] = g["laba"] / g["laba"].sum()
    g["porsi_om"] = g["omzet"] / g["omzet"].sum()
    by_om = g.sort_values("omzet", ascending=False).head(2)
    by_lb = g.sort_values("laba", ascending=False)
    by_lb = by_lb[~by_lb["dept"].isin(by_om["dept"])].head(2)
    return dict(om=by_om, lb=by_lb, om_share=by_om["porsi_om"].sum(), om_laba=by_om["porsi_laba"].sum(),
                lb_share=by_lb["porsi_om"].sum(), lb_laba=by_lb["porsi_laba"].sum())
