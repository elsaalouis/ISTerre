"""
08f_confidence_threshold_selective_classification.py
=======================================================
ISTerre internship — Environmental seismology in glaciology
Author : Elsa Louis
Date   : August 2026

Goal
----
08e's paired random-sample comparison showed HGB's top-confidence
earthquake/regional calls didn't hold up visually as well as expected --
odd, since HGB is on average the MORE confident model (0.65-0.77 vs the
CNN's 0.44-0.52, see 5.1). That mismatch (high confidence, not-so-reliable)
suggests HGB's confidence score doesn't track visual plausibility as well as
the CNN's does. This script tests the natural follow-up hypothesis: maybe
confidence itself (not the choice of model) is the real lever -- i.e. below
some confidence level, BOTH models are essentially guessing among classes on
near-indistinguishable content, and restricting to high-confidence windows
should make the predicted-class mix look more plausible.

This is exactly "selective classification" / "classification with a reject
option": instead of forcing a label on every window, abstain below a
confidence threshold, trading fewer classified windows for higher reliability
among the ones you keep. This script does NOT modify any classifier or
retrain anything -- it just RE-READS existing predictions_<month>.csv files
(09a and 09b's Phase 2 output, nothing new to run) and recomputes the
predicted-class distribution restricted to increasingly confident subsets, so
the shift (or lack of it) is a number, not an impression from a few paired
figures.

Two views of "confidence threshold", both produced:

  1. FIXED raw-probability thresholds (FIXED_THRESHOLDS_HGB/CNN, e.g. "HGB
     p>=0.7") -- gives you a direct, quotable "at p>=0.7, the earthquake
     share is X%" sentence per model. NOT comparable model-to-model at the
     same raw p value, because HGB's confidence runs systematically higher
     than the CNN's (see the module docstring above and the per-class means
     this script prints in Section 3) -- a HGB p=0.7 and a CNN p=0.7 do not
     represent the same amount of "certainty" relative to that model's own
     scale.

  2. COVERAGE-based thresholds (COVERAGE_LEVELS, e.g. "keep the top 50% most
     confident windows, whatever raw p that takes") -- this is the "quantile
     equivalent" fix: each model's threshold is picked from its OWN
     confidence distribution, so "keep the top 50%" means the same thing
     (half the windows, the more confident half) for both models even though
     the raw p values implementing that differ. THIS is the fair way to put
     HGB and CNN side by side on the same plot -- a risk-coverage curve per
     class: as you get more selective (moving right, lower coverage), does
     the predicted-class share converge to something more plausible, and
     does it converge similarly or differently for HGB vs CNN?

Reading the output
-------------------
If the "it's confidence, not the model" hypothesis is right, the earthquake
(and/or regional) share for HGB should move toward a more plausible value as
coverage drops (i.e. as low-confidence windows get excluded) -- ideally
converging toward something closer to the CNN's own high-confidence-subset
share. If it DOESN'T move much even at low coverage, that argues against
"it's just confidence" and back toward "HGB's score itself is miscalibrated
relative to what it's actually seeing" -- also a defensible, reportable
finding, just a different one. Either way this gives a number to put next to
the qualitative 08e observation, per the "deux methodes d'observation
independantes" logic: pairwise visual check + confidence stratification
pointing the same direction is a much stronger claim than either alone.

Needs nothing but the predictions CSVs -- no SDS, no FDSN, no images, no
cluster access required. Run it anywhere (your machine or the cluster) that
has the predictions_<month>.csv files for both pipelines.

Output layout
-------------
  OUTPUT_DIR/run_YYYYMMDD_HHMMSS/
      class_distribution_by_fixed_threshold.csv   <- one row per (model, threshold)
      class_distribution_by_coverage.csv          <- one row per (model, coverage level)
      fig_confidence_coverage_curves_<stamp>.png  <- risk-coverage curve, CLASSES_OF_INTEREST_FOR_PLOT
      run.log                                     <- also has the per-class mean-confidence
                                                       printout and the headline before/after
                                                       numbers for the report
"""



# =============================================================================
# SECTION 1 — CONFIGURATION
# =============================================================================

# -- Input: one or more predictions_<month>.csv file(s) per model (pooled
# together, same convention as 08d/08e -- so you can cover several months at
# once if you want). ------------------------------------------------------------
HGB_PREDICTIONS_CSVS = [
    "/data/failles/louisels/project/results/outputs_09b/run_20260821_173942/predictions_2025-01.csv",
]
CNN_PREDICTIONS_CSVS = [
    "/data/failles/louisels/project/results/outputs_09a/predictions_2025-01.csv",
]

# -- Optional station restriction (None = every station present) ----------------
STATIONS_FILTER = None   # e.g. ["STA1", "STA2"]

CLASS_ORDER  = ["earthquake", "regional", "rockslide", "ice quake", "noise"]
CLASS_COLORS = {"earthquake": "#1f77b4", "rockslide": "#d62728",
                "ice quake": "#2ca02c", "noise": "#7f7f7f", "regional": "#9467bd"}

# -- Coverage sweep (risk-coverage curve, see module docstring) -- % of windows
# kept, ranked by EACH model's own confidence, highest first. 100 = no filter.
COVERAGE_LEVELS = [100, 95, 90, 85, 80, 75, 70, 65, 60, 55, 50,
                    45, 40, 35, 30, 25, 20, 15, 10, 5]

# -- Fixed raw-probability thresholds -- for quotable "at HGB p>=0.7 ..."
# sentences. Kept as two SEPARATE lists on purpose (see module docstring --
# not directly comparable model-to-model at the same raw value).
FIXED_THRESHOLDS_HGB = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
FIXED_THRESHOLDS_CNN = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

# -- Classes highlighted in the summary plot (the ones flagged as suspicious
# in 08e's paired comparison) -- every class still gets its own column in
# both output CSVs regardless of this list. -------------------------------------
CLASSES_OF_INTEREST_FOR_PLOT = ["earthquake", "regional"]

# -- Reference coverage level used for the printed "before -> after" headline
# numbers in Section 6 (must be one of COVERAGE_LEVELS) -------------------------
REFERENCE_COVERAGE_PCT = 50

OUTPUT_DIR = "/data/failles/louisels/project/results/outputs_08f"



# =============================================================================
# SECTION 2 — SETUP
# =============================================================================

import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_setup import create_run_dir, setup_logging, set_matplotlib_defaults

if REFERENCE_COVERAGE_PCT not in COVERAGE_LEVELS:
    print(f"[ERROR] REFERENCE_COVERAGE_PCT ({REFERENCE_COVERAGE_PCT}) must be one of "
          f"COVERAGE_LEVELS {COVERAGE_LEVELS}.")
    sys.exit(1)

RUN_DIR, STAMP = create_run_dir(OUTPUT_DIR)
log_file, log_path = setup_logging(
    RUN_DIR, "08f_confidence_threshold_selective_classification.py",
    extra_info=(f"{len(HGB_PREDICTIONS_CSVS)} HGB file(s)  |  "
                f"{len(CNN_PREDICTIONS_CSVS)} CNN file(s)  |  "
                f"{len(COVERAGE_LEVELS)} coverage level(s)")
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
set_matplotlib_defaults()



# =============================================================================
# SECTION 3 — LOAD PREDICTIONS + PER-ROW CONFIDENCE
# =============================================================================

def _load_predictions(csv_paths, model_label):
    frames = []
    for fpath in csv_paths:
        if not os.path.isfile(fpath):
            print(f"  [WARN] {model_label}: file not found, skipping: {fpath}")
            continue
        df = pd.read_csv(fpath, low_memory=False)
        df["_source_file"] = os.path.basename(fpath)
        print(f"  [OK] {model_label}: {os.path.basename(fpath)}  {len(df):,} row(s)")
        frames.append(df)
    if not frames:
        print(f"  [ERROR] {model_label}: no predictions file could be read.")
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    if STATIONS_FILTER is not None and "station" in df.columns:
        n_before = len(df)
        df = df[df["station"].isin(STATIONS_FILTER)].copy()
        print(f"  {model_label}: STATIONS_FILTER kept {len(df):,} / {n_before:,} row(s)")
    return df


def _add_winning_proba(df, model_label):
    """
    Per-row confidence = proba_<predicted_class> -- the probability the model
    assigned to the class it actually output ("how sure was it about its own
    guess"). Vectorized per class, same proba_<class> column convention as
    08d/08e/09a/09b.
    """
    if df.empty:
        return df
    df = df.copy()
    df["_winning_proba"] = np.nan
    for c in CLASS_ORDER:
        col = f"proba_{c.replace(' ', '_')}"
        if col not in df.columns:
            continue
        m = df["predicted_class"] == c
        df.loc[m, "_winning_proba"] = df.loc[m, col].values
    n_missing = int(df["_winning_proba"].isna().sum())
    if n_missing:
        print(f"  [WARN] {model_label}: {n_missing:,} / {len(df):,} row(s) have no matching "
              f"proba_<class> column for their predicted_class -- excluded below.")
    return df.dropna(subset=["_winning_proba"]).copy()


print(f"\n{'='*70}")
print(f"  LOADING PREDICTIONS")
print(f"{'='*70}")
df_hgb = _add_winning_proba(_load_predictions(HGB_PREDICTIONS_CSVS, "HGB"), "HGB")
df_cnn = _add_winning_proba(_load_predictions(CNN_PREDICTIONS_CSVS, "CNN"), "CNN")

if df_hgb.empty and df_cnn.empty:
    print("[ERROR] Nothing loaded for either model -- check the CSV paths in Section 1.")
    log_file.close()
    sys.exit(1)

print(f"\n{'='*70}")
print(f"  CONFIDENCE OVERVIEW  (grounds the '0.65-0.77 vs 0.44-0.52' -type numbers in data)")
print(f"{'='*70}")
for label, df in (("HGB", df_hgb), ("CNN", df_cnn)):
    if df.empty:
        print(f"\n  {label}: no data loaded, skipped.")
        continue
    print(f"\n  {label}: {len(df):,} window(s) total")
    print(f"    overall winning-proba   mean={df['_winning_proba'].mean():.3f}  "
          f"median={df['_winning_proba'].median():.3f}  "
          f"min={df['_winning_proba'].min():.3f}  max={df['_winning_proba'].max():.3f}")
    for c in CLASS_ORDER:
        sub = df.loc[df["predicted_class"] == c, "_winning_proba"]
        if len(sub) == 0:
            continue
        print(f"      {c:<11s} n={len(sub):6,d}  mean_conf={sub.mean():.3f}  "
              f"median_conf={sub.median():.3f}")



# =============================================================================
# SECTION 4 — CLASS DISTRIBUTION AT A GIVEN CONFIDENCE CUT
# =============================================================================

def _class_distribution(df_kept, n_total):
    n_kept = len(df_kept)
    row = {
        "n_kept":   n_kept,
        "n_total":  n_total,
        "pct_kept": 100.0 * n_kept / n_total if n_total else np.nan,
    }
    vc = df_kept["predicted_class"].value_counts() if n_kept else pd.Series(dtype=int)
    for c in CLASS_ORDER:
        n_c = int(vc.get(c, 0))
        row[f"n_{c.replace(' ', '_')}"]   = n_c
        row[f"pct_{c.replace(' ', '_')}"] = 100.0 * n_c / n_kept if n_kept else np.nan
    return row


def _fixed_threshold_table(df, thresholds, model_label):
    if df.empty:
        return pd.DataFrame()
    n_total = len(df)
    rows = []
    for thr in thresholds:
        kept = df[df["_winning_proba"] >= thr]
        row = {"model": model_label, "threshold": thr}
        row.update(_class_distribution(kept, n_total))
        rows.append(row)
    return pd.DataFrame(rows)


def _coverage_threshold_table(df, coverage_levels, model_label):
    """
    For each target coverage %, pick the threshold from THIS model's OWN
    confidence distribution (its (100-coverage)-th percentile) so "top X%
    most confident" means the same thing -- a fraction of ITS OWN windows --
    for HGB and CNN alike, even though the raw probability implementing that
    differs between them (see module docstring).
    """
    if df.empty:
        return pd.DataFrame()
    n_total = len(df)
    rows = []
    for cov_pct in coverage_levels:
        thr  = float(np.percentile(df["_winning_proba"], 100 - cov_pct))
        kept = df[df["_winning_proba"] >= thr]
        row = {"model": model_label, "coverage_target_pct": cov_pct, "implied_threshold": thr}
        row.update(_class_distribution(kept, n_total))
        rows.append(row)
    return pd.DataFrame(rows)


print(f"\n{'='*70}")
print(f"  COMPUTING CLASS DISTRIBUTIONS")
print(f"{'='*70}")

fixed_hgb = _fixed_threshold_table(df_hgb, FIXED_THRESHOLDS_HGB, "HGB")
fixed_cnn = _fixed_threshold_table(df_cnn, FIXED_THRESHOLDS_CNN, "CNN")
fixed_all = pd.concat([fixed_hgb, fixed_cnn], ignore_index=True)
fixed_path = os.path.join(RUN_DIR, "class_distribution_by_fixed_threshold.csv")
fixed_all.to_csv(fixed_path, index=False)
print(f"  [SAVED] {os.path.basename(fixed_path)}  ({len(fixed_all)} row(s))")

cov_hgb = _coverage_threshold_table(df_hgb, COVERAGE_LEVELS, "HGB")
cov_cnn = _coverage_threshold_table(df_cnn, COVERAGE_LEVELS, "CNN")
cov_all = pd.concat([cov_hgb, cov_cnn], ignore_index=True)
cov_path = os.path.join(RUN_DIR, "class_distribution_by_coverage.csv")
cov_all.to_csv(cov_path, index=False)
print(f"  [SAVED] {os.path.basename(cov_path)}  ({len(cov_all)} row(s))")



# =============================================================================
# SECTION 5 — RISK-COVERAGE CURVE PLOT
# =============================================================================

if not cov_all.empty:
    interest = [c for c in CLASSES_OF_INTEREST_FOR_PLOT if c in CLASS_ORDER]
    if not interest:
        print("\n  [WARN] CLASSES_OF_INTEREST_FOR_PLOT is empty or has no valid class -- skipping plot.")
    else:
        model_colors = {"HGB": "#d62728", "CNN": "#1f77b4"}
        fig, axes = plt.subplots(1, len(interest), figsize=(6.5 * len(interest), 5), squeeze=False)
        axes = axes[0]
        for ax, cls in zip(axes, interest):
            pct_col = f"pct_{cls.replace(' ', '_')}"
            for model_label in ("HGB", "CNN"):
                sub = cov_all[cov_all["model"] == model_label].sort_values("pct_kept")
                if sub.empty or pct_col not in sub.columns:
                    continue
                ax.plot(sub["pct_kept"], sub[pct_col], marker="o", markersize=4,
                        color=model_colors[model_label], label=model_label, linewidth=1.8)
            ax.set_xlabel("% of windows kept (coverage)", fontweight="bold")
            ax.set_ylabel(f"% predicted '{cls}' among kept windows", fontweight="bold")
            ax.set_title(f"{cls}", fontweight="bold")
            ax.invert_xaxis()   # 100% (no filter) on the left, most selective on the right
            ax.grid(True, alpha=0.3)
            ax.legend()
        fig.suptitle("Selective classification: predicted-class share vs. confidence-based "
                      "coverage\n(does restricting to high-confidence windows change the mix?)",
                      fontweight="bold")
        plt.tight_layout()
        fig_path = os.path.join(RUN_DIR, f"fig_confidence_coverage_curves_{STAMP}.png")
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n  [SAVED] {os.path.basename(fig_path)}")



# =============================================================================
# SECTION 6 — HEADLINE NUMBERS FOR THE REPORT
# =============================================================================

print(f"\n{'='*70}")
print(f"  SUMMARY -- headline before/after numbers")
print(f"  (all windows  ->  top {REFERENCE_COVERAGE_PCT}% most confident, per model)")
print(f"{'='*70}")

for model_label, df_cov in (("HGB", cov_hgb), ("CNN", cov_cnn)):
    if df_cov.empty:
        continue
    base = df_cov[df_cov["coverage_target_pct"] == 100]
    ref  = df_cov[df_cov["coverage_target_pct"] == REFERENCE_COVERAGE_PCT]
    if base.empty or ref.empty:
        continue
    base, ref = base.iloc[0], ref.iloc[0]
    print(f"\n  {model_label}  (threshold at {REFERENCE_COVERAGE_PCT}% coverage = "
          f"p>={ref['implied_threshold']:.3f}, n={int(ref['n_kept']):,}/{int(ref['n_total']):,}):")
    for c in CLASS_ORDER:
        col = f"pct_{c.replace(' ', '_')}"
        print(f"    {c:<11s} {base[col]:5.1f}%  (all windows)  ->  {ref[col]:5.1f}%  "
              f"(top {REFERENCE_COVERAGE_PCT}% most confident)")

print(f"\n{'='*70}")
print(f"  DONE")
print(f"{'='*70}")
print(f"  Run folder: {RUN_DIR}")

log_file.close()
