"""Grafik interaktif (Plotly)."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def support_distribution(sup: pd.Series, names: dict, top: int = 25):
    d = sup.head(top).rename("support").reset_index(); d.columns = ["item", "support"]
    d["label"] = d["item"].map(lambda x: f"{names.get(x, x)} ({x})")
    fig = px.bar(d.iloc[::-1], x="support", y="label", orientation="h", text=d["support"].iloc[::-1].map(lambda v: f"{v:.2%}"))
    fig.update_layout(height=max(350, 18 * len(d)), xaxis_tickformat=".1%", yaxis_title="", xaxis_title="Support item",
                      margin=dict(l=10, r=10, t=10, b=10))
    return fig


def support_ccdf(sup: pd.Series):
    """Berapa item yang support-nya >= ambang tertentu (membantu memilih rentang min support)."""
    s = sup.sort_values(ascending=False).reset_index(drop=True)
    d = pd.DataFrame({"ambang": s.values, "jumlah_item": range(1, len(s) + 1)})
    fig = px.line(d, x="ambang", y="jumlah_item", log_x=True, log_y=True)
    for t in [0.001, 0.005, 0.01, 0.05]:
        fig.add_vline(x=t, line_dash="dot", line_color="grey", annotation_text=f"{t*100:g}%")
    fig.update_layout(xaxis_title="Ambang support (log)", yaxis_title="Jumlah item dengan support ≥ ambang (log)",
                      height=380, margin=dict(l=10, r=10, t=10, b=10))
    return fig


def basket_size_hist(X: pd.DataFrame):
    size = X.sum(axis=1)
    fig = px.histogram(size, nbins=int(size.max()))
    fig.update_layout(showlegend=False, xaxis_title="Jumlah item per transaksi", yaxis_title="Jumlah transaksi",
                      height=320, margin=dict(l=10, r=10, t=10, b=10))
    return fig


def rules_heatmap(res: pd.DataFrame, value: str = "aturan", algo: str = "FP-Growth"):
    d = res[res["algoritma"] == algo].pivot_table(index="min_confidence", columns="min_support", values=value)
    fig = go.Figure(go.Heatmap(z=d.values, x=[f"{c*100:g}%" for c in d.columns], y=[f"{c*100:g}%" for c in d.index],
                               text=d.round(2).values, texttemplate="%{text}", colorscale="Blues"))
    fig.update_layout(xaxis_title="Minimum support", yaxis_title="Minimum confidence", height=380,
                      margin=dict(l=10, r=10, t=10, b=10))
    fig.update_xaxes(type="category"); fig.update_yaxes(type="category")
    return fig


def response_curve(res: pd.DataFrame, x: str, y: str, color: str, fixed: dict, log_y=False):
    d = res.copy()
    for k, v in fixed.items():
        d = d[d[k] == v]
    fig = px.line(d.sort_values(x), x=x, y=y, color=color, markers=True, log_x=(x == "min_support"), log_y=log_y)
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10))
    if x in ("min_support", "min_confidence"):
        fig.update_xaxes(tickformat=".1%")
    return fig


def compute_curves(res: pd.DataFrame, conf: float):
    d = res[res["min_confidence"] == conf]
    f1 = px.line(d, x="min_support", y="waktu_ms", color="algoritma", markers=True, log_x=True, log_y=True)
    f1.update_layout(yaxis_title="Waktu median (ms)", xaxis_title="Minimum support", height=360, margin=dict(l=10, r=10, t=30, b=10),
                     title="Waktu eksekusi")
    f1.update_xaxes(tickformat=".1%")
    f2 = px.line(d.dropna(subset=["memori_mb"]), x="min_support", y="memori_mb", color="algoritma", markers=True,
                 log_x=True, log_y=True)
    f2.update_layout(yaxis_title="Memori puncak (MB)", xaxis_title="Minimum support", height=360,
                     margin=dict(l=10, r=10, t=30, b=10), title="Memori puncak")
    f2.update_xaxes(tickformat=".1%")
    return f1, f2


def rule_scatter(r: pd.DataFrame):
    d = r.copy()
    d["aturan"] = d["antecedents_txt"] + " → " + d["consequents_txt"]
    d["status"] = d["robust"].map({True: "Robust", False: "Tidak robust"}) if "robust" in d else "Aturan"
    fig = px.scatter(d, x="support", y="confidence", size="lift", color="status", hover_name="aturan",
                     hover_data={"lift": ":.2f", "support": ":.4f", "confidence": ":.2f"},
                     color_discrete_map={"Robust": "#1f77b4", "Tidak robust": "#bbbbbb"})
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
    fig.update_xaxes(tickformat=".2%"); fig.update_yaxes(tickformat=".0%")
    return fig
