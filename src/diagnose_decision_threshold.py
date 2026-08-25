"""
diagnose_decision_threshold.py
================================
Follow-up to diagnose_09b_predictions.py / diagnose_train_vs_continuous_shift.py.

Context
-------
After the 04a units-bug fix and retrain, 09b's continuous-data classification
stopped collapsing (all 5 classes now get real probability mass) but earthquake
still dominates the predicted-class distribution (69.2% of January 2025,
63.2% of August 2025 detections) — visually inconsistent with the chapter-2
analysis, while ice quake/rockslide review-gallery examples look correct.
The mean-probability confusion matrix from diagnose_09b_predictions.py showed
earthquake sitting at 17-21% mean probability as RUNNER-UP on almost every
other predicted class, which suggests at least part of this is a genuinely
close, low-margin call for the model rather than a confident (but wrong) one
-- exactly the kind of thing a decision-margin/probability-floor rule on TOP
of the existing argmax can catch, without touching the model itself.

This script is PURELY a post-processing diagnostic on the predictions_<month>.csv
files 09b's Phase 2b/2c already wrote -- no SDS/FDSN/sklearn needed, so it
runs fine while the FDSN outage is ongoing (and even off the cluster, e.g.
straight from OneDrive on a laptop).

What it does
------------
1. Loads every CSV in PREDICTIONS_CSVS, and for every row computes:
     - winning class + probability (== 09b's current naive-argmax decision)
     - runner-up class + probability
     - margin = winning_proba - runnerup_proba
2. Reports how "borderline" the earthquake calls specifically are: the
   margin distribution for rows currently predicted earthquake, and what
   class is usually the runner-up when the margin is small.
3. Sweeps two alternative decision rules against the sweep thresholds below
   and reports how the predicted-class distribution (especially earthquake's
   share) shifts under each, plus what fraction of windows get reclassified
   as "uncertain" as a result:
     (a) GLOBAL PROBABILITY FLOOR — require winning_proba >= floor, else
         "uncertain"
     (b) GLOBAL MARGIN REQUIREMENT — require margin >= margin_threshold,
         else "uncertain"
     (c) EARTHQUAKE-ONLY MARGIN REQUIREMENT — same as (b) but ONLY applied
         when the winning class is earthquake (every other class keeps the
         naive argmax decision) -- the most targeted test of "is earthquake
         specifically winning a lot of close calls".
4. Saves:
     - predictions_enriched_<source>.csv   (original rows + margin/runner-up
       columns, one per input file, next to a copy of the original filename)
     - threshold_sweep_summary.csv          (one row per rule x threshold x
       source, class counts + % uncertain)
     - fig_threshold_sweep.png              (earthquake share & % uncertain
       vs threshold, for rules b/c)

This does NOT retrain or touch the model -- it only tells you whether
earthquake's over-prediction is fixable by calibration alone, or whether it
survives even a strict margin/floor requirement (in which case it's more
likely a genuine model/feature-space limitation -- see 06c's new
EARTHQUAKE_REBALANCE_MODE experiment and 08d's larger review gallery for the
two other angles on this).

Usage
-----
    python3 diagnose_decision_threshold.py [csv1 csv2 ...]

With no arguments, uses the PREDICTIONS_CSVS default list below (Elsa's
January/August 2025 run). Works with any number of predictions_<month>.csv
files, from any 09b run.
"""

import os
import sys

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except ImportError:
    _HAVE_MPL = False


# =============================================================================
# CONFIGURATION
# =============================================================================

PREDICTIONS_CSVS = [
    r"C:\Users\elsa.louis\OneDrive - ESTIA\Documents\4 ISTERRE\project\results\09b_continuous_data_test\run_20260821_173942\predictions_2025-01.csv",
    r"C:\Users\elsa.louis\OneDrive - ESTIA\Documents\4 ISTERRE\project\results\09b_continuous_data_test\run_20260821_173942\predictions_2025-08.csv",
]

CLASS_ORDER = ["earthquake", "regional", "rockslide", "ice quake", "noise"]
TARGET_CLASS = "earthquake"   # the class under investigation -- everything in
                              # Section 3/4 is framed around this one, but the
                              # global rules (a)/(b) apply to all classes too

OUTPUT_DIR = r"C:\Users\elsa.louis\OneDrive - ESTIA\Documents\4 ISTERRE\project\results\09b_continuous_data_test\run_20260821_173942\diagnose_threshold"   # where predictions_enriched_*.csv / summary CSV / figure land
                    # defaults to the current directory -- override to a real
                    # path if you want outputs somewhere specific

# ── Threshold sweeps ────────────────────────────────────────────────────────
PROBA_FLOOR_THRESHOLDS  = [0.0, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
MARGIN_THRESHOLDS        = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40]

# Bins used for the "how borderline are earthquake's calls" margin histogram
MARGIN_HIST_BINS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.01]


# =============================================================================
# HELPERS
# =============================================================================

def proba_cols_for(class_order):
    return [f"proba_{c.replace(' ', '_')}" for c in class_order]


def enrich(df, class_order):
    """
    Add winning/runner-up class + probability + margin columns to df.
    Uses the proba_* columns directly (NOT df['predicted_class']) so this is
    self-consistent even if predicted_class was written by a differently-
    ordered class list upstream.
    """
    pcols = proba_cols_for(class_order)
    missing = [c for c in pcols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing probability column(s): {missing}. "
                          f"Does CLASS_ORDER match this CSV's classes?")

    proba = df[pcols].values.astype(float)
    order_desc = np.argsort(-proba, axis=1)   # descending, per row

    win_idx = order_desc[:, 0]
    run_idx = order_desc[:, 1]

    class_arr = np.array(class_order)
    df = df.copy()
    df["_win_class"]    = class_arr[win_idx]
    df["_win_proba"]    = proba[np.arange(len(df)), win_idx]
    df["_runnerup_class"] = class_arr[run_idx]
    df["_runnerup_proba"] = proba[np.arange(len(df)), run_idx]
    df["_margin"]        = df["_win_proba"] - df["_runnerup_proba"]
    return df


def class_distribution(labels, class_order, extra_label="uncertain"):
    """Return an ordered dict-like Series of counts for class_order + extra_label."""
    vc = pd.Series(labels).value_counts()
    out = {c: int(vc.get(c, 0)) for c in class_order}
    out[extra_label] = int(vc.get(extra_label, 0))
    return out


def apply_floor_rule(df, floor):
    """Rule (a): winning_proba >= floor, else 'uncertain'. Applies to ALL classes."""
    decision = df["_win_class"].where(df["_win_proba"] >= floor, "uncertain")
    return decision


def apply_margin_rule(df, margin_thr, only_class=None):
    """
    Rule (b)/(c): margin >= margin_thr, else 'uncertain'.
    If only_class is given, the requirement is applied ONLY to rows whose
    winning class is only_class -- every other row keeps its naive argmax
    decision unconditionally (rule c). only_class=None applies it globally
    (rule b).
    """
    if only_class is None:
        fails = df["_margin"] < margin_thr
    else:
        fails = (df["_win_class"] == only_class) & (df["_margin"] < margin_thr)
    decision = df["_win_class"].where(~fails, "uncertain")
    return decision


def pct(n, total):
    return 100.0 * n / total if total else float("nan")


# =============================================================================
# MAIN
# =============================================================================

def main():
    csv_paths = sys.argv[1:] if len(sys.argv) > 1 else PREDICTIONS_CSVS

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    frames = []
    for path in csv_paths:
        if not os.path.exists(path):
            print(f"[WARN] Not found, skipping: {path}")
            continue
        df = pd.read_csv(path, low_memory=False)
        if df.empty:
            print(f"[WARN] Empty, skipping: {path}")
            continue
        source = os.path.splitext(os.path.basename(path))[0]
        df = enrich(df, CLASS_ORDER)
        df["_source"] = source
        frames.append(df)

        enriched_path = os.path.join(OUTPUT_DIR, f"predictions_enriched_{source}.csv")
        keep_cols = [c for c in df.columns if not c.startswith("_")] + [
            "_win_class", "_win_proba", "_runnerup_class", "_runnerup_proba", "_margin",
        ]
        df[keep_cols].rename(columns={
            "_win_class": "winning_class", "_win_proba": "winning_proba",
            "_runnerup_class": "runnerup_class", "_runnerup_proba": "runnerup_proba",
            "_margin": "margin",
        }).to_csv(enriched_path, index=False)
        print(f"[OK] {source}: {len(df):,} rows loaded, enriched CSV -> {enriched_path}")

    if not frames:
        print("[ERROR] No usable predictions CSVs found. Nothing to do.")
        sys.exit(1)

    df_all = pd.concat(frames, ignore_index=True)
    df_all["_source"] = "ALL"
    sources = frames + [df_all]

    # =========================================================================
    print(f"\n{'='*74}")
    print("  SECTION 1 — Baseline (naive argmax, current 09b behavior)")
    print(f"{'='*74}")
    for df in sources:
        src = df["_source"].iloc[0]
        total = len(df)
        dist = class_distribution(df["_win_class"], CLASS_ORDER, extra_label=None)
        del dist[None]
        print(f"\n  --- {src}  (n={total:,}) ---")
        for c in CLASS_ORDER:
            n = dist[c]
            print(f"    {c:<14s} {n:8,d}  ({pct(n, total):5.1f}%)")

    # =========================================================================
    print(f"\n{'='*74}")
    print(f"  SECTION 2 — How borderline are '{TARGET_CLASS}' calls specifically?")
    print(f"  (margin = winning proba - runner-up proba, among rows predicted "
          f"{TARGET_CLASS})")
    print(f"{'='*74}")
    for df in sources:
        src = df["_source"].iloc[0]
        eq = df[df["_win_class"] == TARGET_CLASS]
        if eq.empty:
            print(f"\n  --- {src}: no '{TARGET_CLASS}' predictions ---")
            continue
        print(f"\n  --- {src}  ({TARGET_CLASS} n={len(eq):,}) ---")
        print(f"    margin  min={eq['_margin'].min():.3f}  median={eq['_margin'].median():.3f}  "
              f"mean={eq['_margin'].mean():.3f}  max={eq['_margin'].max():.3f}")
        counts, edges = np.histogram(eq["_margin"], bins=MARGIN_HIST_BINS)
        for lo, hi, n in zip(edges[:-1], edges[1:], counts):
            hi_disp = "1.00" if hi > 1 else f"{hi:.2f}"
            print(f"      margin in [{lo:.2f}, {hi_disp})  {n:7,d}  ({pct(n, len(eq)):5.1f}%)")
        ru_counts = eq.loc[eq["_margin"] < 0.10, "_runnerup_class"].value_counts()
        n_close = int((eq["_margin"] < 0.10).sum())
        print(f"    Of the {n_close:,} ({pct(n_close, len(eq)):.1f}%) '{TARGET_CLASS}' calls "
              f"with margin < 0.10, runner-up class breakdown:")
        for cls, n in ru_counts.items():
            print(f"      {cls:<14s} {n:7,d}  ({pct(n, n_close):5.1f}% of close calls)")

    # =========================================================================
    print(f"\n{'='*74}")
    print("  SECTION 3 — Rule (a): global probability floor")
    print("  winning_proba >= floor, else -> 'uncertain'. Applies to ALL classes.")
    print(f"{'='*74}")
    summary_rows = []
    for df in sources:
        src = df["_source"].iloc[0]
        total = len(df)
        print(f"\n  --- {src} ---")
        header = "    floor  " + "".join(f"{c[:4]:>8s}" for c in CLASS_ORDER) + f"{'uncertain':>11s}"
        print(header)
        for floor in PROBA_FLOOR_THRESHOLDS:
            decision = apply_floor_rule(df, floor)
            dist = class_distribution(decision, CLASS_ORDER)
            row_str = f"    {floor:5.2f}  " + "".join(f"{dist[c]:8,d}" for c in CLASS_ORDER) \
                       + f"{dist['uncertain']:11,d}"
            print(row_str)
            summary_rows.append({
                "source": src, "rule": "proba_floor", "threshold": floor,
                "n_total": total, "n_uncertain": dist["uncertain"],
                "pct_uncertain": round(pct(dist["uncertain"], total), 2),
                **{f"n_{c.replace(' ', '_')}": dist[c] for c in CLASS_ORDER},
                f"pct_{TARGET_CLASS.replace(' ', '_')}": round(pct(dist[TARGET_CLASS], total), 2),
            })

    # =========================================================================
    print(f"\n{'='*74}")
    print("  SECTION 4 — Rule (b): global margin requirement")
    print("  margin >= threshold, else -> 'uncertain'. Applies to ALL classes.")
    print(f"{'='*74}")
    for df in sources:
        src = df["_source"].iloc[0]
        total = len(df)
        print(f"\n  --- {src} ---")
        header = "    margin " + "".join(f"{c[:4]:>8s}" for c in CLASS_ORDER) + f"{'uncertain':>11s}"
        print(header)
        for m in MARGIN_THRESHOLDS:
            decision = apply_margin_rule(df, m, only_class=None)
            dist = class_distribution(decision, CLASS_ORDER)
            row_str = f"    {m:5.2f}  " + "".join(f"{dist[c]:8,d}" for c in CLASS_ORDER) \
                       + f"{dist['uncertain']:11,d}"
            print(row_str)
            summary_rows.append({
                "source": src, "rule": "margin_global", "threshold": m,
                "n_total": total, "n_uncertain": dist["uncertain"],
                "pct_uncertain": round(pct(dist["uncertain"], total), 2),
                **{f"n_{c.replace(' ', '_')}": dist[c] for c in CLASS_ORDER},
                f"pct_{TARGET_CLASS.replace(' ', '_')}": round(pct(dist[TARGET_CLASS], total), 2),
            })

    # =========================================================================
    print(f"\n{'='*74}")
    print(f"  SECTION 5 — Rule (c): margin requirement, '{TARGET_CLASS}' ONLY")
    print(f"  every other class keeps its naive argmax decision unconditionally --")
    print(f"  the most targeted test of whether {TARGET_CLASS} specifically is")
    print(f"  winning a lot of close calls.")
    print(f"{'='*74}")
    for df in sources:
        src = df["_source"].iloc[0]
        total = len(df)
        print(f"\n  --- {src} ---")
        header = "    margin " + "".join(f"{c[:4]:>8s}" for c in CLASS_ORDER) + f"{'uncertain':>11s}"
        print(header)
        for m in MARGIN_THRESHOLDS:
            decision = apply_margin_rule(df, m, only_class=TARGET_CLASS)
            dist = class_distribution(decision, CLASS_ORDER)
            row_str = f"    {m:5.2f}  " + "".join(f"{dist[c]:8,d}" for c in CLASS_ORDER) \
                       + f"{dist['uncertain']:11,d}"
            print(row_str)
            summary_rows.append({
                "source": src, "rule": f"margin_{TARGET_CLASS.replace(' ', '_')}_only",
                "threshold": m,
                "n_total": total, "n_uncertain": dist["uncertain"],
                "pct_uncertain": round(pct(dist["uncertain"], total), 2),
                **{f"n_{c.replace(' ', '_')}": dist[c] for c in CLASS_ORDER},
                f"pct_{TARGET_CLASS.replace(' ', '_')}": round(pct(dist[TARGET_CLASS], total), 2),
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(OUTPUT_DIR, "threshold_sweep_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n[SAVED] {summary_path}")

    # =========================================================================
    # Figure: earthquake share & % uncertain vs threshold, rules (b)/(c), ALL source
    # =========================================================================
    if _HAVE_MPL:
        all_summary = summary_df[summary_df["source"] == "ALL"]
        eq_col = f"pct_{TARGET_CLASS.replace(' ', '_')}"

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

        ax = axes[0]
        for rule, label, color in [
            ("margin_global", "margin (all classes)", "#1f77b4"),
            (f"margin_{TARGET_CLASS.replace(' ', '_')}_only", f"margin ({TARGET_CLASS} only)", "#d62728"),
        ]:
            sub = all_summary[all_summary["rule"] == rule].sort_values("threshold")
            if not sub.empty:
                ax.plot(sub["threshold"], sub[eq_col], marker="o", label=label, color=color)
        sub_floor = all_summary[all_summary["rule"] == "proba_floor"].sort_values("threshold")
        if not sub_floor.empty:
            ax.plot(sub_floor["threshold"], sub_floor[eq_col], marker="o",
                    label="probability floor (all classes)", color="#2ca02c")
        base_pct = all_summary.loc[
            (all_summary["rule"] == "margin_global") & (all_summary["threshold"] == 0.0), eq_col
        ]
        if not base_pct.empty:
            ax.axhline(base_pct.iloc[0], color="gray", linestyle="--", linewidth=1,
                       label="baseline (naive argmax)")
        ax.set_xlabel("Threshold")
        ax.set_ylabel(f"{TARGET_CLASS} share of all predictions (%)")
        ax.set_title(f"{TARGET_CLASS.title()} share vs decision threshold")
        ax.legend(fontsize=8)

        ax = axes[1]
        for rule, label, color in [
            ("margin_global", "margin (all classes)", "#1f77b4"),
            (f"margin_{TARGET_CLASS.replace(' ', '_')}_only", f"margin ({TARGET_CLASS} only)", "#d62728"),
            ("proba_floor", "probability floor (all classes)", "#2ca02c"),
        ]:
            sub = all_summary[all_summary["rule"] == rule].sort_values("threshold")
            if not sub.empty:
                ax.plot(sub["threshold"], sub["pct_uncertain"], marker="o", label=label, color=color)
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Windows reclassified 'uncertain' (%)")
        ax.set_title("Coverage cost of each rule")
        ax.legend(fontsize=8)

        fig.suptitle("Decision-threshold recalibration — post-processing only, no retrain",
                     fontsize=11, fontweight="bold")
        fig.tight_layout()
        fig_path = os.path.join(OUTPUT_DIR, "fig_threshold_sweep.png")
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print(f"[SAVED] {fig_path}")
    else:
        print("[INFO] matplotlib not available -- skipped fig_threshold_sweep.png "
              "(the printed tables and threshold_sweep_summary.csv still have everything).")

    print(f"\n{'='*74}")
    print("  How to read this")
    print(f"{'='*74}")
    print(f"  If {TARGET_CLASS}'s share drops sharply and stays down as the margin/floor")
    print(f"  threshold increases (Section 4/5 tables, or the left figure panel), most of")
    print(f"  its over-prediction is coming from CLOSE calls -- fixable by calibration")
    print(f"  alone, no retrain needed. If it stays high even at strict thresholds (e.g.")
    print(f"  margin >= 0.30) while 'uncertain' balloons for every class equally, the")
    print(f"  model is CONFIDENTLY choosing {TARGET_CLASS} too often -- that's the signature")
    print(f"  of a genuine representational limit (broader learned decision region from")
    print(f"  more real training diversity, not a calibration artifact) and points instead")
    print(f"  toward 06c's EARTHQUAKE_REBALANCE_MODE retrain experiment or a direct")
    print(f"  09a/09b comparison.")


if __name__ == "__main__":
    main()
