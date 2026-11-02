"""
Seleksi aturan robust: signifikansi (Fisher + Benjamini-Hochberg), stabilitas temporal (dua paruh),
dan stabilitas subsampel (80% tanpa pengembalian).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

from .mining import fpgrowth, rules_from, to_matrix


def _key(a, c):
    return (frozenset(a), frozenset(c))


def _rules(items: pd.DataFrame, s: float, c: float, max_len: int | None):
    X = to_matrix(items)
    fi = fpgrowth(X, min_support=s, use_colnames=True, max_len=max_len)
    return X, rules_from(fi, len(X), c)


def fisher_bh(X: pd.DataFrame, rules: pd.DataFrame) -> pd.DataFrame:
    """Uji eksak Fisher satu sisi (asosiasi positif) + koreksi BH."""
    r = rules.copy()
    if not len(r):
        return r.assign(n_bersama=[], p_fisher=[], q_bh=[])
    pv, nn = [], []
    for a, c in zip(r["antecedents"], r["consequents"]):
        A = X[list(a)].all(axis=1).values; C = X[list(c)].all(axis=1).values
        tab = [[int((A & C).sum()), int((A & ~C).sum())], [int((~A & C).sum()), int((~A & ~C).sum())]]
        pv.append(fisher_exact(tab, alternative="greater")[1]); nn.append(tab[0][0])
    r["n_bersama"] = nn; r["p_fisher"] = pv
    r["q_bh"] = multipletests(pv, method="fdr_bh")[1]
    return r


def temporal_split(items: pd.DataFrame, s: float, c: float, max_len: int | None, how: str = "rentang"):
    """Bagi data jadi dua paruh berdasarkan tanggal.
    how='rentang' : batas = tengah rentang tanggal kalender (dibulatkan ke awal hari) — sama dengan artikel
                    (1-15 vs 16-31 Maret);
    how='median'  : batas = median waktu transaksi (dua paruh berjumlah transaksi hampir sama)."""
    d = items.dropna(subset=["date"])
    if d.empty:
        return None
    tdate = d.groupby("trx")["date"].min()
    if how == "median":
        cut = tdate.sort_values().iloc[len(tdate) // 2]
    else:
        lo, hi = tdate.min().normalize(), tdate.max().normalize()
        cut = lo + pd.Timedelta(days=((hi - lo).days + 1) // 2)
    h1 = tdate[tdate < cut].index; h2 = tdate[tdate >= cut].index
    _, r1 = _rules(items[items["trx"].isin(h1)], s, c, max_len)
    _, r2 = _rules(items[items["trx"].isin(h2)], s, c, max_len)
    k1 = set(_key(a, b) for a, b in zip(r1.antecedents, r1.consequents))
    k2 = set(_key(a, b) for a, b in zip(r2.antecedents, r2.consequents))
    return dict(cut=cut, n1=len(h1), n2=len(h2), k1=k1, k2=k2)


def subsample_recurrence(items: pd.DataFrame, s: float, c: float, B: int = 100, frac: float = 0.8,
                         seed: int = 42, max_len: int | None = 4, progress=None) -> dict:
    """Proporsi subsampel (tanpa pengembalian) yang memunculkan tiap aturan.
    Bootstrap dengan pengembalian sengaja tidak dipakai: pada ambang sangat rendah, satu keranjang besar
    yang terambil berulang membuat seluruh subset-nya frekuen dan jumlah pola meledak."""
    rng = np.random.default_rng(seed)
    trx = items["trx"].unique()
    g = items.groupby("trx")["item"].apply(lambda x: sorted(set(x)))
    cnt: dict = {}
    for i in range(B):
        if progress: progress((i + 1) / B, f"Subsampel {i+1}/{B}")
        pick = rng.choice(trx, int(frac * len(trx)), replace=False)
        sub = g.loc[pick].explode().reset_index(); sub.columns = ["trx", "item"]
        _, r = _rules(sub, s, c, max_len)
        for k in set(_key(a, b) for a, b in zip(r.antecedents, r.consequents)):
            cnt[k] = cnt.get(k, 0) + 1
    return {k: v / B for k, v in cnt.items()}


def robust_rules(items: pd.DataFrame, s: float, c: float, B: int = 100, frac: float = 0.8, seed: int = 42,
                 alpha: float = 0.05, min_recur: float = 0.5, max_len: int | None = None,
                 use_temporal: bool = True, progress=None, split_how: str = "rentang") -> tuple[pd.DataFrame, dict]:
    X, r = _rules(items, s, c, max_len)
    r = fisher_bh(X, r)
    info = dict(n_trx=len(X), n_rules=len(r))
    if not len(r):
        return r, info
    keys = [_key(a, b) for a, b in zip(r.antecedents, r.consequents)]
    r["signifikan"] = r["q_bh"] < alpha
    if use_temporal:
        ts = temporal_split(items, s, c, max_len, split_how)
        if ts:
            r["paruh_1"] = [k in ts["k1"] for k in keys]; r["paruh_2"] = [k in ts["k2"] for k in keys]
            r["dua_paruh"] = r["paruh_1"] & r["paruh_2"]
            info.update(batas_paruh=str(ts["cut"])[:10], n_paruh1=ts["n1"], n_paruh2=ts["n2"])
        else:
            r["dua_paruh"] = True; info["catatan"] = "Tidak ada kolom tanggal: uji temporal dilewati."
    else:
        r["dua_paruh"] = True
    rec = subsample_recurrence(items, s, c, B, frac, seed, max_len or 4, progress)
    r["rekurensi"] = [rec.get(k, 0.0) for k in keys]
    r["robust"] = r["signifikan"] & r["dua_paruh"] & (r["rekurensi"] >= min_recur)
    info.update(signifikan=int(r["signifikan"].sum()), dua_paruh=int(r["dua_paruh"].sum()),
                rekurensi_ok=int((r["rekurensi"] >= min_recur).sum()), robust=int(r["robust"].sum()))
    return r, info
