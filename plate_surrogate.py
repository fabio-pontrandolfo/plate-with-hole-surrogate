"""Surrogate models for peak stress in a plate with a central hole.

Labels come from a closed-form formula (Heywood's stress concentration factor)
plus 1 % random noise standing in for mesh error, so the true answer is known
everywhere. That makes it a clean sandbox for the surrogate workflow:
design of experiments, model comparison, and an extrapolation test.

Run:  python plate_surrogate.py      -> prints results, writes results/
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import qmc  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.gaussian_process import GaussianProcessRegressor  # noqa: E402
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C  # noqa: E402
from sklearn.metrics import mean_absolute_error, r2_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.neural_network import MLPRegressor  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"
rng = np.random.default_rng(0)


def fake_fea(X):
    """'Solver': W width, d hole dia, t thickness [mm], F load [N] -> peak stress [MPa]"""
    W, d, t, F = X.T
    Kt = 2 + (1 - d / W) ** 3                 # Heywood approximation
    s = Kt * F / ((W - d) * t)
    return s * (1 + 0.01 * rng.standard_normal(len(s)))   # 1% 'mesh noise'


def rel_err(p, y):
    return np.abs(p - y) / y


def main():
    OUT.mkdir(exist_ok=True)
    results = {}

    # 1) Design of Experiments: Latin hypercube inside the design envelope
    lo, hi = [40, 5, 2, 5_000], [100, 30, 6, 20_000]
    X = qmc.scale(qmc.LatinHypercube(d=4, seed=0).random(200), lo, hi)
    y = fake_fea(X)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)

    models = {
        "GaussianProcess": make_pipeline(StandardScaler(), GaussianProcessRegressor(
            C(1.0) * RBF([1, 1, 1, 1]) + WhiteKernel(1e-3), normalize_y=True, n_restarts_optimizer=3)),
        "RandomForest": RandomForestRegressor(300, random_state=0),
        "MLP": make_pipeline(StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=5000, random_state=0)),
    }
    # log-transform the target: stresses span a wide range
    preds = {}
    for name, m in models.items():
        m.fit(X_tr, np.log(y_tr))
        p = np.exp(m.predict(X_te))
        preds[name] = p
        e = rel_err(p, y_te)
        results[name] = dict(r2=r2_score(y_te, p), mae_mpa=mean_absolute_error(y_te, p),
                             mean_err_pct=100 * e.mean(), max_err_pct=100 * e.max())
        print(f"{name:16s} R2={r2_score(y_te, p):.3f}  MAE={mean_absolute_error(y_te, p):6.1f} MPa  "
              f"max err={e.max() * 100:5.1f}%")

    # 2) Extrapolation test: thinner plates than anything in training
    X_out = qmc.scale(qmc.LatinHypercube(d=4, seed=1).random(50), [40, 5, 1, 5_000], [100, 30, 1.8, 20_000])
    y_out = fake_fea(X_out)
    gp, rf = models["GaussianProcess"], models["RandomForest"]
    p_in, s_in = gp.predict(X_te, return_std=True)
    p_out, s_out = gp.predict(X_out, return_std=True)
    e_in, e_out = rel_err(np.exp(p_in), y_te), rel_err(np.exp(p_out), y_out)
    e_rf_out = rel_err(np.exp(rf.predict(X_out)), y_out)
    # share of true values inside the GP's 95 % interval (in log space)
    cov_in = np.mean(np.abs(np.log(y_te) - p_in) < 1.96 * s_in)
    cov_out = np.mean(np.abs(np.log(y_out) - p_out) < 1.96 * s_out)
    results["extrapolation"] = dict(
        gp_in_mean_err_pct=100 * e_in.mean(), gp_out_mean_err_pct=100 * e_out.mean(),
        rf_out_mean_err_pct=100 * e_rf_out.mean(),
        gp_in_mean_std_pct=100 * s_in.mean(), gp_out_mean_std_pct=100 * s_out.mean(),
        gp_95pct_coverage_in=cov_in, gp_95pct_coverage_out=cov_out)
    print(f"GP in-envelope:  mean err {100 * e_in.mean():.1f}%, std {100 * s_in.mean():.1f}%, "
          f"95% interval covers {100 * cov_in:.0f}%")
    print(f"GP out-of-range: mean err {100 * e_out.mean():.1f}%, std {100 * s_out.mean():.1f}%, "
          f"95% interval covers {100 * cov_out:.0f}%")
    print(f"RF out-of-range: mean err {100 * e_rf_out.mean():.1f}%")

    (OUT / "results.json").write_text(json.dumps(results, indent=2, default=float))
    plot(y_te, preds, X_te, X_out, y_out, gp, rf)


def plot(y_te, preds, X_te, X_out, y_out, gp, rf):
    colors = ["#2a78d6", "#eb6834", "#1baf7a"]
    ink2, grid, surface = "#52514e", "#e4e3df", "#fcfcfb"
    plt.rcParams.update({
        "figure.facecolor": surface, "axes.facecolor": surface, "axes.edgecolor": ink2,
        "xtick.color": ink2, "ytick.color": ink2, "axes.grid": True, "grid.color": grid,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "savefig.dpi": 150, "savefig.bbox": "tight"})

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    for k, (name, p) in enumerate(preds.items()):
        ax.plot(y_te, p, "o", ms=4, alpha=0.75, mec="none", color=colors[k], label=name)
    lim = [y_te.min() * 0.9, y_te.max() * 1.05]
    ax.plot(lim, lim, color=ink2, lw=1)
    ax.set_xlabel("true peak stress (MPa)")
    ax.set_ylabel("predicted (MPa)")
    ax.set_title("In range: the GP sits on the line")
    ax.legend(fontsize=8)

    ax = axes[1]
    for X_, y_ in ((X_te, y_te), (X_out, y_out)):
        t = X_[:, 2]
        ax.plot(t, 100 * (np.exp(gp.predict(X_)) / y_ - 1), "o", ms=4, alpha=0.75, mec="none",
                color=colors[0], label="GaussianProcess" if X_ is X_te else None)
        ax.plot(t, 100 * (np.exp(rf.predict(X_)) / y_ - 1), "o", ms=4, alpha=0.75, mec="none",
                color=colors[1], label="RandomForest" if X_ is X_te else None)
    ax.axvline(2.0, color=ink2, lw=1, ls="--")
    ax.axhline(0, color=ink2, lw=0.8)
    ax.text(2.1, ax.get_ylim()[0] * 0.9, "training range →", color=ink2, fontsize=8)
    ax.set_xlabel("plate thickness t (mm)")
    ax.set_ylabel("prediction error (%)")
    ax.set_title("Thinner than training: both under-predict stress")
    ax.legend(fontsize=8)
    fig.savefig(OUT / "surrogate_results.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
