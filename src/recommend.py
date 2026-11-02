"""Penamaan item pada aturan dan teks rekomendasi promosi."""
from __future__ import annotations

import pandas as pd


def label_rules(r: pd.DataFrame, names: dict, show_code: bool = False) -> pd.DataFrame:
    def fmt(s):
        return " + ".join(f"{names.get(x, x)}" + (f" ({x})" if show_code else "") for x in sorted(s))
    d = r.copy()
    d["antecedents_txt"] = d["antecedents"].map(fmt)
    d["consequents_txt"] = d["consequents"].map(fmt)
    return d


def tidy_table(d: pd.DataFrame) -> pd.DataFrame:
    """Tabel rapi untuk ditampilkan/diunduh."""
    cols = {"antecedents_txt": "Antesedan", "consequents_txt": "Konsekuen", "n_bersama": "n transaksi",
            "support": "Support", "confidence": "Confidence", "lift": "Lift", "q_bh": "q (BH)",
            "dua_paruh": "Dua paruh", "rekurensi": "Rekurensi", "robust": "Robust"}
    keep = [c for c in cols if c in d.columns]
    return d[keep].rename(columns=cols).sort_values("Lift", ascending=False).reset_index(drop=True)


def recommendation_text(d: pd.DataFrame, level: str, max_items: int = 15) -> list[str]:
    """Kalimat rekomendasi untuk aturan robust (atau semua aturan bila uji robust belum dijalankan)."""
    use = d[d["robust"]] if "robust" in d.columns else d
    use = use.sort_values(["lift", "confidence"], ascending=False).head(max_items)
    out = []
    for _, r in use.iterrows():
        n = int(r["n_bersama"]) if "n_bersama" in r and pd.notna(r["n_bersama"]) else None
        bukti = f"{n} transaksi" if n is not None else f"support {r['support']:.2%}"
        if level == "Departemen":
            aksi = "letakkan raknya berdekatan atau buat display gabungan"
        else:
            aksi = "jadikan paket bundling atau tawarkan sebagai produk pelengkap di kasir"
        caution = " Bukti tipis — konfirmasi ke pemilik toko." if (n is not None and n < 10) else ""
        out.append(f"**{r['antecedents_txt']} → {r['consequents_txt']}** — {r['confidence']:.0%} pembeli antesedan "
                   f"juga membeli konsekuen (lift {r['lift']:.1f}; {bukti}). Saran: {aksi}.{caution}")
    return out
