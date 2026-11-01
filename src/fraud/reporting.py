"""Tables and figures for the imbalance comparison (shared by the script and the notebook)."""
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import precision_recall_curve


def build_tables(out):
    res = out["results"]
    rank, ops = [], []
    for name, r in res.items():
        m = r["at_0.5"]
        lo, hi = r["ap_ci"]; dlo, dhi = r["ap_diff_vs_unweighted"]
        rank.append({"strategy": name, "PR-AUC": round(m["pr_auc"], 3),
                     "95% CI": f"{lo:.3f} to {hi:.3f}",
                     "diff vs unweighted (95% CI)": f"{dlo:+.3f} to {dhi:+.3f}",
                     "recall@90%prec": round(m["recall_at_90_precision"], 3),
                     "ROC-AUC": round(m["roc_auc"], 3),
                     "mean score": round(r["mean_score_test"], 4),
                     "Brier": round(r["brier_test"], 5)})
        for label, key in [("0.5", "at_0.5"), ("val-tuned", "at_val_tuned")]:
            k = r[key]
            ops.append({"strategy": name, "threshold": label, "thr value": round(k["threshold"], 3),
                        "precision": round(k["precision"], 3), "recall": round(k["recall"], 3),
                        "F1": round(k["f1"], 3), "TP": k["tp"], "FP": k["fp"], "FN": k["fn"]})
    return pd.DataFrame(rank), pd.DataFrame(ops)


def make_figures(out, fig_dir):
    res, y = out["results"], out["y_test"]
    fig, ax = plt.subplots(figsize=(7, 4.8))
    for name, s in out["scores_test"].items():
        p, r, _ = precision_recall_curve(y, s)
        ax.plot(r, p, label=f"{name} ({res[name]['at_0.5']['pr_auc']:.3f})")
    ax.axhline(y.mean(), ls="--", c="grey", label=f"random ({y.mean():.4f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("Imbalance strategies: PR curves (test)")
    ax.legend(); fig.tight_layout(); fig.savefig(fig_dir / "imbalance_pr_curves.png", dpi=120); plt.close(fig)

    names = list(res)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ap = [res[n]["at_0.5"]["pr_auc"] for n in names]
    err = [[res[n]["at_0.5"]["pr_auc"] - res[n]["ap_ci"][0] for n in names],
           [res[n]["ap_ci"][1] - res[n]["at_0.5"]["pr_auc"] for n in names]]
    axes[0].bar(names, ap, yerr=err, capsize=4); axes[0].set_title("PR-AUC with 95% bootstrap CI")
    axes[0].tick_params(axis="x", rotation=20)
    axes[1].bar(names, [res[n]["mean_score_test"] for n in names])
    axes[1].axhline(y.mean(), ls="--", c="grey", label="true fraud rate"); axes[1].set_yscale("log")
    axes[1].set_title("Mean predicted score (calibration check)"); axes[1].legend()
    axes[1].tick_params(axis="x", rotation=20)
    fig.tight_layout(); fig.savefig(fig_dir / "imbalance_summary.png", dpi=120); plt.close(fig)


# ----------------------------------------------------------------------------- step 3
def gbt_tables(out):
    """Return (ranking, costs, sensitivity, calibration) DataFrames for the GBT experiment."""
    rank = []
    for name, r in out["ranking"].items():
        b = r["alert_budget"]
        rank.append({"model": name, "PR-AUC": round(r["pr_auc"], 3),
                     "95% CI": f"{r['ci'][0]:.3f} to {r['ci'][1]:.3f}",
                     "diff vs LR (95% CI)": f"{r['diff_vs_lr'][0]:+.3f} to {r['diff_vs_lr'][1]:+.3f}",
                     "recall@90%prec": round(r["recall_at_90_precision"], 3),
                     f"recall @ top {b['k']} alerts": round(b["recall"], 3),
                     f"precision @ top {b['k']} alerts": round(b["precision"], 3),
                     "ROC-AUC": round(r["roc_auc"], 3)})
    costs = pd.DataFrame([{"rule": r["rule"], "threshold": None if r["threshold"] is None else round(r["threshold"], 4),
                           "flagged": r["flagged"], "TP": r["tp"], "FP": r["fp"], "FN": r["fn"],
                           "missed $": round(r["missed_amount"], 2), "review $": round(r["review_cost"], 2),
                           "total cost $": round(r["cost"], 2), "savings vs nothing": f"{r['savings_pct']:.1f}%"}
                          for r in out["cost_rows"]])
    sens = out["sensitivity"].copy()
    sens["threshold"] = sens["threshold"].round(4); sens["cost"] = sens["cost"].round(2)
    sens["no_model_cost"] = sens["no_model_cost"].round(2); sens["savings_pct"] = sens["savings_pct"].round(1)
    c = out["calibration"]
    cal = pd.DataFrame({"mean score": {"raw": c["raw"]["mean_score"], "calibrated": c["calibrated"]["mean_score"],
                                       "true fraud rate": c["true_rate"]},
                        "Brier": {"raw": c["raw"]["brier"], "calibrated": c["calibrated"]["brier"],
                                  "true fraud rate": float("nan")}}).round(5)
    return pd.DataFrame(rank), costs, sens, cal


def gbt_figures(out, fig_dir):
    import numpy as np
    from .cost import cost_curve
    y, amt, c_fp = out["y_test"], out["amt_test"], out["c_fp"]
    fig, ax = plt.subplots(figsize=(7, 4.8))
    for name in ["Logistic regression", "LightGBM"]:
        p, r, _ = precision_recall_curve(y, out["scores_test"][name])
        ax.plot(r, p, label=f"{name} ({out['ranking'][name]['pr_auc']:.3f})")
    ax.axhline(y.mean(), ls="--", c="grey", label=f"random ({y.mean():.4f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("LightGBM vs logistic regression (test)")
    ax.legend(); fig.tight_layout(); fig.savefig(fig_dir / "gbt_pr_curves.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for split, y_, a_, s_, ls in [("validation", out["y_val"], out["amt_val"], out["scores_val"]["LightGBM"], "-"),
                                  ("test", y, amt, out["scores_test"]["LightGBM"], "--")]:
        thr, costs = cost_curve(y_, s_, a_, c_fp)
        m = np.isfinite(thr)
        ax.plot(thr[m], costs[m], ls, label=f"{split} (cost of flagging nothing = ${costs[0]:,.0f})")
    ax.axvline(out["thresholds"]["cost"], c="grey", ls=":", label=f"cost-tuned threshold {out['thresholds']['cost']:.3f}")
    ax.set_xscale("log"); ax.set_xlabel("Threshold (log scale)"); ax.set_ylabel("Total cost ($)")
    ax.set_title(f"Total cost vs threshold (false alarm = ${c_fp:g})"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(fig_dir / "gbt_cost_curve.png", dpi=120); plt.close(fig)

    s = out["sensitivity"]
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.bar(s["c_fp"].astype(str), s["savings_pct"]); ax.axhline(0, c="black", lw=0.8)
    ax.set_xlabel("Assumed cost of a false alarm ($)"); ax.set_ylabel("Savings vs flagging nothing (%)")
    ax.set_title("Sensitivity to the false-alarm cost")
    fig.tight_layout(); fig.savefig(fig_dir / "gbt_sensitivity.png", dpi=120); plt.close(fig)
