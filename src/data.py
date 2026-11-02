"""
Pemuatan dan pembersihan data transaksi.

Dua format masukan didukung:
1. Log POS (format toko): file CSV ber-delimiter ';' dengan kolom FLAG, CODE, DESC, QTY, PRICE, NOTA, TGL.
   Aturan pembersihan mengikuti artikel:
     - hanya baris FLAG == 'PLU' (item terjual),
     - hanya nota yang memiliki baris 'TTL' (transaksi selesai),
     - item = kode PLU (CODE), bukan deskripsi,
     - item ganda dalam satu nota dihitung sekali (data biner).
2. Format umum: CSV apa pun dengan minimal kolom ID transaksi dan kolom item
   (opsional: kolom deskripsi dan kolom tanggal).
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class CleanResult:
    """Hasil pembersihan: tabel item bersih + log langkah pembersihan."""
    items: pd.DataFrame                 # kolom: trx, item, desc, date (date boleh NaT)
    steps: list[dict] = field(default_factory=list)
    checks: dict = field(default_factory=dict)


def read_csv_auto(raw: bytes) -> pd.DataFrame:
    """Baca CSV dengan menebak delimiter (',' / ';' / tab). Semua kolom dibaca sebagai teks."""
    text = raw.decode("utf-8", errors="replace")
    head = text[:5000]
    sep = max([";", ",", "\t"], key=head.count)
    df = pd.read_csv(io.StringIO(text), sep=sep, dtype=str, na_values=["NULL", "null", ""], keep_default_na=True)
    df.columns = [c.strip() for c in df.columns]
    return df


def is_pos_log(df: pd.DataFrame) -> bool:
    """True bila struktur kolom cocok dengan log POS toko."""
    need = {"FLAG", "CODE", "NOTA", "TGL"}
    return need.issubset(set(df.columns))


def clean_pos_log(df: pd.DataFrame) -> CleanResult:
    """Pembersihan log POS sesuai Tabel 1 artikel."""
    steps = [dict(tahap="0", aturan="Data mentah", baris=len(df), transaksi=df["NOTA"].nunique())]
    checks: dict = {"flag_counts": df["FLAG"].value_counts().to_dict()}

    plu = df[df["FLAG"] == "PLU"].copy()
    steps.append(dict(tahap="1", aturan="Ambil baris PLU saja", baris=len(plu), transaksi=plu["NOTA"].nunique()))

    ttl_notas = set(df.loc[df["FLAG"] == "TTL", "NOTA"])
    dropped = sorted(set(plu["NOTA"]) - ttl_notas)
    plu = plu[plu["NOTA"].isin(ttl_notas)]
    steps.append(dict(tahap="2", aturan="Pertahankan nota ber-TTL (transaksi selesai)", baris=len(plu),
                      transaksi=plu["NOTA"].nunique()))
    checks["nota_tanpa_ttl"] = dropped

    # Pemeriksaan: apakah TTL = sum(QTY*PRICE) baris PLU? (menentukan perlu/tidaknya mengurangi VD1)
    if {"QTY", "PRICE"}.issubset(df.columns):
        q = pd.to_numeric(plu["QTY"], errors="coerce"); pr = pd.to_numeric(plu["PRICE"], errors="coerce")
        s = (q * pr).groupby(plu["NOTA"]).sum()
        t = pd.to_numeric(df.loc[df["FLAG"] == "TTL", "PRICE"], errors="coerce").groupby(df.loc[df["FLAG"] == "TTL", "NOTA"]).sum()
        both = pd.concat([s, t], axis=1, keys=["plu", "ttl"]).dropna()
        checks["ttl_sama_dengan_plu"] = float((both["plu"] == both["ttl"]).mean()) if len(both) else None

    # Konsistensi kode vs deskripsi
    if "DESC" in plu.columns:
        checks["kode_multi_deskripsi"] = int((plu.groupby("CODE")["DESC"].nunique() > 1).sum())
        checks["deskripsi_multi_kode"] = int((plu.groupby("DESC")["CODE"].nunique() > 1).sum())

    dup = int(plu.duplicated(["NOTA", "CODE"]).sum())
    items = pd.DataFrame({
        "trx": plu["NOTA"].values,
        "item": plu["CODE"].values,
        "desc": plu["DESC"].values if "DESC" in plu.columns else plu["CODE"].values,
        "date": pd.to_datetime(plu["TGL"], errors="coerce").values,
    }).drop_duplicates(["trx", "item"])
    steps.append(dict(tahap="3", aturan="Item = kode PLU; item ganda per nota dihitung sekali",
                      baris=len(items), transaksi=items["trx"].nunique()))
    checks["duplikat_item_per_nota"] = dup
    return CleanResult(items=items.reset_index(drop=True), steps=steps, checks=checks)


def clean_generic(df: pd.DataFrame, trx_col: str, item_col: str,
                  desc_col: str | None = None, date_col: str | None = None) -> CleanResult:
    """Pembersihan untuk format umum: buang baris kosong, item ganda per transaksi dihitung sekali."""
    steps = [dict(tahap="0", aturan="Data mentah", baris=len(df), transaksi=df[trx_col].nunique())]
    d = df.dropna(subset=[trx_col, item_col]).copy()
    steps.append(dict(tahap="1", aturan="Buang baris tanpa ID transaksi/item", baris=len(d), transaksi=d[trx_col].nunique()))
    items = pd.DataFrame({
        "trx": d[trx_col].astype(str).str.strip().values,
        "item": d[item_col].astype(str).str.strip().values,
        "desc": (d[desc_col] if desc_col else d[item_col]).astype(str).values,
        "date": pd.to_datetime(d[date_col], errors="coerce").values if date_col else pd.NaT,
    })
    n0 = len(items)
    items = items.drop_duplicates(["trx", "item"])
    steps.append(dict(tahap="2", aturan="Item ganda per transaksi dihitung sekali", baris=len(items),
                      transaksi=items["trx"].nunique()))
    return CleanResult(items=items.reset_index(drop=True), steps=steps,
                       checks={"duplikat_item_per_transaksi": n0 - len(items)})


def item_names(items: pd.DataFrame) -> dict:
    """Peta kode item -> deskripsi yang paling sering muncul."""
    return (items.groupby("item")["desc"].agg(lambda s: s.value_counts().index[0])).to_dict()


def apply_granularity(items: pd.DataFrame, level: str, prefix_len: int = 3) -> pd.DataFrame:
    """Ubah item ke tingkat granularitas tertentu.
    level: 'SKU' (kode asli) | 'Departemen' (n digit awal kode) | 'Deskripsi' (teks deskripsi).
    """
    d = items.copy()
    if level == "Departemen":
        d["item"] = d["item"].str[:prefix_len]
    elif level == "Deskripsi":
        d["item"] = d["desc"].str.strip().str.upper()
    return d.drop_duplicates(["trx", "item"])


def default_dept_names(items: pd.DataFrame, prefix_len: int = 3, top: int = 3) -> pd.DataFrame:
    """Label sementara departemen dari produk dominan (untuk diedit pengguna)."""
    d = items.copy(); d["dept"] = d["item"].str[:prefix_len]
    g = d.groupby("dept")["desc"].agg(lambda s: " | ".join(s.value_counts().index[:top]))
    n = d.groupby("dept")["trx"].nunique()
    return pd.DataFrame({"kode": g.index, "nama": "", "produk_dominan": g.values, "transaksi": n.values}) \
             .sort_values("transaksi", ascending=False).reset_index(drop=True)
