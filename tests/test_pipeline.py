from pathlib import Path

from src import data as D, mining as M, robust as RB

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "contoh_transaksi.csv"


def _items():
    df = D.read_csv_auto(SAMPLE.read_bytes())
    return D.clean_generic(df, "no_transaksi", "kode_item", "nama_item", "tanggal").items


def test_apriori_equals_fpgrowth():
    X = M.to_matrix(_items())
    res, rs, eq = M.run_grid(X, [0.005, 0.01], [0.1, 0.3], reps=1, measure_memory=False)
    assert eq["itemset_identik"].all() and eq["aturan_identik"].all()


def test_robust_runs():
    r, info = RB.robust_rules(_items(), 0.005, 0.3, B=10)
    assert info["n_rules"] >= 1 and "robust" in r.columns


def test_department_level():
    it = D.apply_granularity(_items(), "Departemen", 3)
    assert it["item"].str.len().max() == 3
