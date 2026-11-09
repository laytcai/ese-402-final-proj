# Baseline checkpoint: mean, least squares, ridge and kNN for superconductor Tc,
# scored with random and grouped 5-fold CV. Run with: python baseline.py

import json
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler

SEED = 4020  # picked before looking at any results
FOLDS = 5
REPEATS = 5
BOOT_DRAWS = 2000
TEST_SHARE = 0.2
LAMBDAS = np.logspace(-3, 6, 37)
KS = np.arange(1, 51)

ROOT = Path(__file__).resolve().parent
ZIP_PATH = ROOT / "data" / "superconductivity.zip"
OUT = ROOT / "results"
DATA_URL = "https://archive.ics.uci.edu/static/public/464/superconductivty+data.zip"


def load_data():
    if not ZIP_PATH.exists():
        ZIP_PATH.parent.mkdir(exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, ZIP_PATH)
    with zipfile.ZipFile(ZIP_PATH) as z:
        inputs = pd.read_csv(z.open("train.csv"))
        formulas = pd.read_csv(z.open("unique_m.csv"))
    y = inputs.pop("critical_temp").to_numpy()
    assert (y == formulas["critical_temp"].to_numpy()).all()  # both files are in the same row order
    names = formulas["material"].to_numpy()

    # rows with identical inputs count as one material, however the formula is spelled
    groups = inputs.groupby(list(inputs.columns), sort=False).ngroup().to_numpy()
    assert pd.Series(groups).groupby(names).nunique().max() == 1  # each formula is inside one group
    return inputs, y, groups, names


def repeat_stats(y, groups):
    tc = pd.Series(y).groupby(groups)
    repeated = (tc.transform("size") > 1).to_numpy()
    gap = y - tc.transform("mean").to_numpy()
    sd = tc.std().dropna()
    return {
        "repeated_rows": int(repeated.sum()),
        "repeated_materials": int(len(sd)),
        "median_tc_sd": float(sd.median()),
        "p90_tc_sd": float(sd.quantile(0.9)),
        "max_tc_sd": float(sd.max()),
        # inputs are identical on these rows, so no model can beat each material's own mean
        "floor_rmse_repeated_rows": float(np.sqrt(np.mean(gap[repeated] ** 2))),
        "floor_rmse_all_rows": float(np.sqrt(np.mean(gap**2))),
    }


def check_against_paper(X, y):
    # Hamidieh (2018): least squares, 25 random 2/3 - 1/3 splits, all rows
    rng = np.random.default_rng([SEED, 0])
    n_test = len(y) // 3
    mses, r2s = [], []
    for _ in range(25):
        order = rng.permutation(len(y))
        test, train = order[:n_test], order[n_test:]
        scaler = StandardScaler().fit(X[train])
        model = LinearRegression().fit(scaler.transform(X[train]), y[train])
        pred = model.predict(scaler.transform(X[test]))
        mses.append(np.mean((pred - y[test]) ** 2))
        r2s.append(np.corrcoef(pred, y[test])[0, 1] ** 2)  # the paper's R2 is squared correlation
    return {
        "rmse": float(np.sqrt(np.mean(mses))),
        "rmse_sd_over_splits": float(np.std(np.sqrt(mses), ddof=1)),
        "r2": float(np.mean(r2s)),
    }


def seal_test_rows(groups):
    rng = np.random.default_rng([SEED, 1])
    shuffled = rng.permutation(np.unique(groups))
    sealed = shuffled[: round(TEST_SHARE * len(shuffled))]
    return np.isin(groups, sealed)


def deal_folds(groups, rng):
    # deal whole groups into folds like cards
    shuffled = rng.permutation(np.unique(groups))
    fold_of_group = np.zeros(groups.max() + 1, dtype=int)
    fold_of_group[shuffled] = np.arange(len(shuffled)) % FOLDS
    return fold_of_group[groups]


def share_with_copy_in_training(groups):
    # under random folds, the share of rows with an identical row on the training side
    shares = []
    for r in range(REPEATS):
        folds = deal_folds(np.arange(len(groups)), np.random.default_rng([SEED, 2, 0, r]))
        folds_per_group = pd.Series(folds).groupby(groups).transform("nunique").to_numpy()
        shares.append((folds_per_group > 1).mean())
    return float(np.mean(shares))


# each model returns predictions for the validation rows, one column per setting
def fit_mean(Xtr, ytr, Xva):
    return np.full((len(Xva), 1), ytr.mean())


def fit_least_squares(Xtr, ytr, Xva):
    return LinearRegression().fit(Xtr, ytr).predict(Xva)[:, None]


def fit_ridge(Xtr, ytr, Xva):
    return np.column_stack([Ridge(alpha=lam).fit(Xtr, ytr).predict(Xva) for lam in LAMBDAS])


def fit_knn(Xtr, ytr, Xva):
    dist = cdist(Xva, Xtr)
    # stable sort so ties between identical rows always break the same way
    nearest = np.argsort(dist, axis=1, kind="stable")[:, : KS.max()]
    return np.cumsum(ytr[nearest], axis=1) / KS  # average of the first k neighbours, for every k


MODELS = {
    "mean": (fit_mean, [np.nan]),
    "least squares": (fit_least_squares, [np.nan]),
    "ridge": (fit_ridge, LAMBDAS),
    "knn": (fit_knn, KS),
}


def out_of_fold(X, y, folds):
    preds = {name: np.zeros((len(y), len(settings))) for name, (_, settings) in MODELS.items()}
    for k in range(FOLDS):
        train, valid = folds != k, folds == k
        scaler = StandardScaler().fit(X[train])  # learned inside the fold only
        Xtr, Xva = scaler.transform(X[train]), scaler.transform(X[valid])
        for name, (fit, _) in MODELS.items():
            preds[name][valid] = fit(Xtr, y[train], Xva)
    return preds


def run_cv(X, y, groups):
    # random folds are the same dealing with every row as its own group
    schemes = {"random": np.arange(len(y)), "grouped": groups}
    preds = {}
    for s, (scheme, g) in enumerate(schemes.items()):
        runs = []
        for r in range(REPEATS):
            folds = deal_folds(g, np.random.default_rng([SEED, 2, s, r]))
            assert pd.Series(folds).groupby(g).nunique().max() == 1
            runs.append(out_of_fold(X, y, folds))
            print(f"  {scheme} repeat {r + 1} done", flush=True)
        for name in MODELS:
            preds[scheme, name] = np.stack([run[name] for run in runs])  # repeats x rows x settings
    return preds


def tune(preds, y):
    # the fold-weighted loss from 4A is the same as the MSE over all out-of-fold rows
    picks, chosen, curves = {}, {}, []
    for (scheme, name), p in preds.items():
        mse = ((p - y[None, :, None]) ** 2).mean(axis=1)  # repeats x settings
        best = mse.mean(axis=0).argmin()
        settings = MODELS[name][1]
        picks[scheme, name] = p[:, :, best]
        chosen[scheme, name] = {
            "setting": settings[best],
            "at_grid_edge": len(settings) > 1 and best in (0, len(settings) - 1),
            "setting_per_repeat": [round(float(settings[i]), 4) for i in mse.argmin(axis=1)],
        }
        rmse = np.sqrt(mse)
        for j, setting in enumerate(settings):
            curves.append(
                {
                    "scheme": scheme,
                    "model": name,
                    "setting": setting,
                    "rmse_mean": rmse[:, j].mean(),
                    "rmse_sd": rmse[:, j].std(ddof=1),
                }
            )
    return picks, chosen, pd.DataFrame(curves)


def bootstrap(picks, y, groups):
    # resample materials, not rows, using the same draws for every model and scheme
    keys = list(picks)
    labels = pd.factorize(groups)[0]
    n_groups = labels.max() + 1
    rows_per_group = np.bincount(labels)
    sse = np.stack(
        [np.bincount(labels, weights=(picks[key][r] - y) ** 2) for key in keys for r in range(REPEATS)],
        axis=1,
    )
    rng = np.random.default_rng([SEED, 3])
    draws = np.zeros((BOOT_DRAWS, len(keys)))
    for b in range(BOOT_DRAWS):
        counts = np.bincount(rng.integers(n_groups, size=n_groups), minlength=n_groups)
        rmse = np.sqrt(counts @ sse / (counts @ rows_per_group))
        draws[b] = rmse.reshape(len(keys), REPEATS).mean(axis=1)
    return {key: draws[:, i] for i, key in enumerate(keys)}


def rmse_per_repeat(pred, y):
    return np.sqrt(((pred - y) ** 2).mean(axis=1))


def summary_table(picks, chosen, draws, y, groups):
    repeated = pd.Series(groups).groupby(groups).transform("size").to_numpy() > 1
    rows = []
    for (scheme, name), pred in picks.items():
        rmse = rmse_per_repeat(pred, y)
        rows.append(
            {
                "scheme": scheme,
                "model": name,
                "setting": chosen[scheme, name]["setting"],
                "at_grid_edge": chosen[scheme, name]["at_grid_edge"],
                "rmse_mean": rmse.mean(),
                "rmse_sd_over_repeats": rmse.std(ddof=1),
                "ci_low": np.percentile(draws[scheme, name], 2.5),
                "ci_high": np.percentile(draws[scheme, name], 97.5),
                "rmse_repeated_rows": rmse_per_repeat(pred[:, repeated], y[repeated]).mean(),
                "setting_per_repeat": chosen[scheme, name]["setting_per_repeat"],
            }
        )
    return pd.DataFrame(rows)


def paired_table(picks, draws, y):
    pairs = []
    for scheme in ("random", "grouped"):
        pairs.append(((scheme, "ridge"), (scheme, "least squares")))
        pairs.append(((scheme, "knn"), (scheme, "ridge")))
    for name in MODELS:
        pairs.append((("grouped", name), ("random", name)))
    rows = []
    for a, b in pairs:
        diff = rmse_per_repeat(picks[a], y) - rmse_per_repeat(picks[b], y)
        boot = draws[a] - draws[b]
        rows.append(
            {
                "a": " ".join(a),
                "b": " ".join(b),
                "diff_mean": diff.mean(),
                "diff_sd_over_repeats": diff.std(ddof=1),
                "ci_low": np.percentile(boot, 2.5),
                "ci_high": np.percentile(boot, 97.5),
            }
        )
    return pd.DataFrame(rows)


def main():
    start = time.time()
    OUT.mkdir(exist_ok=True)
    inputs, y, groups, names = load_data()
    X = inputs.to_numpy()

    spellings = pd.Series(names).groupby(groups).nunique()
    facts = {
        "seed": SEED,
        "rows": len(y),
        "inputs": X.shape[1],
        "formula_names": int(pd.Series(names).nunique()),
        "materials": int(len(np.unique(groups))),
        "materials_with_several_spellings": int((spellings > 1).sum()),
        "rows_in_those_materials": int(np.isin(groups, spellings[spellings > 1].index).sum()),
        "exact_duplicate_rows": int(inputs.assign(tc=y).duplicated().sum()),
        "tc_min": float(y.min()),
        "tc_median": float(np.median(y)),
        "tc_mean": float(y.mean()),
        "tc_sd": float(y.std(ddof=1)),
    }

    # these two steps use all rows, but nothing is tuned or chosen from them
    facts["repeats_all_rows"] = repeat_stats(y, groups)
    facts["paper_check"] = check_against_paper(X, y)
    print(json.dumps(facts, indent=2), flush=True)

    # seal the test set. Everything below only ever sees the other rows
    sealed = seal_test_rows(groups)
    assert not set(groups[sealed]) & set(groups[~sealed])
    facts.update(
        sealed_rows=int(sealed.sum()),
        sealed_materials=int(len(np.unique(groups[sealed]))),
        dev_rows=int((~sealed).sum()),
        dev_materials=int(len(np.unique(groups[~sealed]))),
    )
    X, y, groups = X[~sealed], y[~sealed], groups[~sealed]
    facts["repeats_dev_rows"] = repeat_stats(y, groups)
    facts["share_with_copy_in_training_random_folds"] = share_with_copy_in_training(groups)

    preds = run_cv(X, y, groups)
    picks, chosen, curves = tune(preds, y)
    draws = bootstrap(picks, y, groups)
    summary = summary_table(picks, chosen, draws, y, groups)
    paired = paired_table(picks, draws, y)

    (OUT / "facts.json").write_text(json.dumps(facts, indent=2) + "\n")
    summary.to_csv(OUT / "summary.csv", index=False, float_format="%.4f")
    paired.to_csv(OUT / "paired.csv", index=False, float_format="%.4f")
    curves.to_csv(OUT / "tuning.csv", index=False, float_format="%.4f")

    pd.set_option("display.width", 200, "display.max_columns", 20)
    print(summary.drop(columns="setting_per_repeat").round(3).to_string(index=False))
    print(paired.round(3).to_string(index=False))
    print(f"done in {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
