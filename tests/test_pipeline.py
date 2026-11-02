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


def _clean_sample():
    df = D.read_csv_auto(SAMPLE.read_bytes())
    return D.clean_generic(df, "no_transaksi", "kode_item", "nama_item", "tanggal", "jumlah", "harga", "harga_pokok")


def test_business_kpis():
    from src import business as BZ
    cr = _clean_sample()
    d = BZ.prepare(cr.lines, 3, {})
    k = BZ.kpis(d)
    assert k["transaksi"] == cr.items["trx"].nunique()
    assert abs(k["omzet"] - (d["qty"] * d["price"]).sum()) < 1e-6
    assert 0 < k["margin"] < 1
    assert BZ.has_time(d)
    pk = BZ.peak_hours(d)
    assert pk["blok"][0] < pk["blok"][1]
    prod = BZ.products(d)
    assert BZ.pareto(prod)["n50"] <= BZ.pareto(prod)["n_produk"]


def test_rupiah_format():
    from src import business as BZ
    assert BZ.rp(63357078, True) == "Rp 63,4 jt"
    assert BZ.rp(12500) == "Rp 12.500"
