"""
Penambangan itemset/aturan dan eksperimen sensitivitas Apriori vs FP-Growth.
"""
from __future__ import annotations

import math
import time
import tracemalloc

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder

ALGOS = ("Apriori", "FP-Growth")


def to_matrix(items: pd.DataFrame) -> pd.DataFrame:
    """Matriks transaksi-item biner (baris = transaksi, kolom = item)."""
    baskets = items.groupby("trx")["item"].apply(lambda s: sorted(set(s))).tolist()
    te = TransactionEncoder()
    return pd.DataFrame(te.fit(baskets).transform(baskets), columns=te.columns_)


def profile(X: pd.DataFrame) -> dict:
    """Karakteristik data: ukuran, kepadatan, ukuran keranjang, sebaran support item."""
    size = X.sum(axis=1)
    sup = X.mean(axis=0).sort_values(ascending=False)
    return dict(
        transaksi=int(X.shape[0]), item=int(X.shape[1]),
        kepadatan=float(X.values.mean()),
        rata_item=float(size.mean()), median_item=float(size.median()), maks_item=int(size.max()),
        satu_item=float((size == 1).mean()),
        support_maks=float(sup.iloc[0]),
        item_ge_1=int((sup >= 0.01).sum()), item_ge_05=int((sup >= 0.005).sum()), item_ge_01=int((sup >= 0.001).sum()),
        support_series=sup,
    )


def min_count(min_support: float, n: int) -> int:
    """Jumlah transaksi minimum yang setara dengan ambang support relatif."""
    return int(math.ceil(min_support * n - 1e-9))


def mine(X: pd.DataFrame, algo: str, min_support: float, max_len: int | None = None,
         low_memory: bool = False) -> pd.DataFrame:
    if algo == "Apriori":
        return apriori(X, min_support=min_support, use_colnames=True, max_len=max_len, low_memory=low_memory)
    return fpgrowth(X, min_support=min_support, use_colnames=True, max_len=max_len)


def rules_from(fi: pd.DataFrame, n_trx: int, min_conf: float) -> pd.DataFrame:
    """Bentuk aturan; kembalikan DataFrame kosong bila tidak ada itemset berukuran >= 2."""
    cols = ["antecedents", "consequents", "support", "confidence", "lift", "leverage", "conviction"]
    if len(fi) == 0 or (fi["itemsets"].map(len) >= 2).sum() == 0:
        return pd.DataFrame(columns=cols)
    r = association_rules(fi, num_itemsets=n_trx, metric="confidence", min_threshold=min_conf)
    return r[cols].reset_index(drop=True)


def rule_key(a, c) -> tuple:
    return (frozenset(a), frozenset(c))


def timed(fn, reps: int = 5):
    """Median waktu (detik) dari beberapa ulangan; hasil eksekusi terakhir dikembalikan."""
    ts, out = [], None
    for _ in range(reps):
        t0 = time.perf_counter(); out = fn(); ts.append(time.perf_counter() - t0)
    return out, float(np.median(ts))


def peak_mem(fn) -> float:
    """Memori puncak (MB) satu eksekusi, diukur terpisah dengan tracemalloc."""
    tracemalloc.start()
    try:
        fn(); _, pk = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return pk / 1e6


def run_grid(X: pd.DataFrame, supports: list[float], confs: list[float], reps: int = 5,
             measure_memory: bool = True, low_memory: bool = False, max_len: int | None = None,
             apriori_floor: float | None = None, progress=None) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """
    Jalankan grid min support x min confidence untuk kedua algoritma.

    apriori_floor: Apriori dilewati untuk support < nilai ini (pengaman memori di server kecil).
    Mengembalikan: (tabel hasil, kamus ruleset per konfigurasi, tabel kesamaan keluaran).
    """
    n = len(X)
    rows, rulesets, equal = [], {}, []
    total = len(supports) * len(ALGOS); step = 0
    for s in sorted(supports):
        fis = {}
        for algo in ALGOS:
            step += 1
            if progress: progress(step / total, f"{algo} · support {s*100:g}%")
            if algo == "Apriori" and apriori_floor is not None and s < apriori_floor:
                continue
            f = lambda: mine(X, algo, s, max_len, low_memory)
            fi, t = timed(f, reps)
            mem = peak_mem(f) if measure_memory else np.nan
            fis[algo] = (fi, t, mem)
        if len(fis) == 2:
            a, b = fis["Apriori"][0], fis["FP-Growth"][0]
            equal.append(dict(min_support=s, itemset_identik=set(map(frozenset, a.itemsets)) == set(map(frozenset, b.itemsets))))
        for algo, (fi, t, mem) in fis.items():
            k2 = int((fi["itemsets"].map(len) >= 2).sum()) if len(fi) else 0
            kmax = int(fi["itemsets"].map(len).max()) if len(fi) else 0
            for c in confs:
                r = rules_from(fi, n, c)
                rulesets[(s, c, algo)] = set(rule_key(x, y) for x, y in zip(r.antecedents, r.consequents))
                rows.append(dict(algoritma=algo, min_support=s, min_count=min_count(s, n), min_confidence=c,
                                 itemset=len(fi), itemset_k2=k2, panjang_maks=kmax,
                                 aturan=len(r), aturan_lift_gt1=int((r["lift"] > 1).sum()) if len(r) else 0,
                                 rerata_support=r["support"].mean() if len(r) else np.nan,
                                 rerata_confidence=r["confidence"].mean() if len(r) else np.nan,
                                 rerata_lift=r["lift"].mean() if len(r) else np.nan,
                                 waktu_ms=t * 1000, memori_mb=mem))
    res = pd.DataFrame(rows)
    eq = pd.DataFrame(equal)
    if len(eq):
        eq["aturan_identik"] = [all(rulesets.get((s, c, "Apriori")) == rulesets.get((s, c, "FP-Growth"))
                                    for c in confs) for s in eq["min_support"]]
    return res, rulesets, eq


def delta_table(res: pd.DataFrame, indicator: str = "aturan", algo: str = "FP-Growth") -> pd.DataFrame:
    """Perubahan indikator antar-ambang support berdekatan (persamaan 4 artikel), per min confidence."""
    out = []
    d = res[res["algoritma"] == algo]
    for c, g in d.groupby("min_confidence"):
        g = g.sort_values("min_support")
        prev = None
        for _, r in g.iterrows():
            if prev is not None:
                old, new = prev[indicator], r[indicator]
                if old == 0:
                    ch, kind = new - old, "selisih absolut"
                else:
                    ch, kind = (new - old) / old * 100, "%"
                out.append(dict(min_confidence=c, dari=prev["min_support"], ke=r["min_support"],
                                nilai_lama=old, nilai_baru=new, perubahan=ch, satuan=kind))
            prev = r
    return pd.DataFrame(out)
