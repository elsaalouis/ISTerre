# Environmental seismology — microseismicity detection & classification

Internship project (ISTerre, April-August 2026) — automatic detection and classification of microseismic events (local earthquakes, rockslides, ice quakes, regional earthquakes, local noise) around the Mont Blanc massif, using two independent classification approaches: 
- a tabular HistGradientBoosting classifier on hand-crafted seismic features
- a CNN classifier on spectrogram images

Author: Elsa Louis — Supervisors: J. Kokowski, E. Larose (ISTerre)

## Contents

- [Setup](#setup)
- [Repository layout](#repository-layout)
- [Shared modules](#shared-modules-not-run-directly)
- [The pipeline](#the-pipeline)
  - [Stage 1 — Event windowing & feature extraction](#stage-1--event-windowing--feature-extraction-build-the-5-classes)
  - [Stage 2 — Classification (two branches)](#stage-2--classification-two-independent-branches)
  - [Stage 3 — Operational validation on continuous data](#stage-3--operational-validation-on-continuous-data)
- [Calibration / one-time analysis scripts](#calibration--one-time-analysis-scripts)
- [Report figures](#report-figures)
- [Early exploration / not part of the current pipeline](#early-exploration--not-part-of-the-current-pipeline)
- [How the scripts chain together](#how-the-scripts-chain-together)

## Setup

The conda environment used for all cluster-side scripts is defined in `src/environment.yml` (env name `glacier-seismo`):

```bash
conda env create -f src/environment.yml
conda activate glacier-seismo
```

Most scripts need two data sources, reachable only from the ISTerre cluster / OSUG VPN:

- **Catalog + picks**: ISTerre FDSN server, `http://ist-sc3-geobs.osug.fr:8080`
- **Waveforms**: ISTerre SDS archive, `/data/sig/SDS`

A few scripts (04c) query public catalogs (EMSC) over the internet. CNN (07c) and DeepDenoiser (03c) trainings run on Google Colab (GPU free access), not on the cluster, and need TensorFlow.

## Repository layout

Every script follows the same internal structure: 
- a module docstring (goal, data sources, pipeline steps, output files)
- a `SECTION 1 — CONFIGURATION` block at the top with every parameter need to change
- then the processing code
**To run a script, open it, edit the `CONFIGURATION` section, and run it directly** (`python3 04a_sta_lta_catalog_windowing.py`)

## Shared modules (not run directly)

| Module | Purpose |
|---|---|
| `catalog_helpers.py` | FDSN catalog queries (by box or by distance), pick/station helpers |
| `preprocessing.py` | Waveform loading, instrument response removal, filtering |
| `run_setup.py` | Run-folder creation, logging, SDS/FDSN connections, inventory fetch |
| `detection.py` | STA/LTA, SNR metrics, kurtosis onset refinement, window merging |
| `features.py` | The 99/103 Maggi–Hibert feature names + safe extraction wrapper |
| `visualization.py` | All plotting functions used across the pipeline |
| `denoiser_utils.py` | Shared helper to run DeepDenoiser inference (used by 03c) |

Third-party code, **not modified** by this project:
- `seismic_params.py` (Maggi/Hibert feature computation — A. Maggi, C. Hibert, E. Pirot, C. Groult)
- `detecteurV3_fonctions.py` (Groult et al. 2026 spectrogram detector)
- `deepdenoiser/` folder (Zhu et al. 2019, DeepDenoiser)

## The pipeline

Two classification methods are trained and validated in parallel, but they share the same upstream data-preparation stage:

```
                     ┌─ Stage 1: event windowing & feature extraction (04a/04c/04d, + 03c/03d rescue)  ─┐
                     │                                                                                  │
                     ▼                                                                                  ▼
      Stage 2a: tabular (06a/06b/06c)                                                     Stage 2b: CNN (07a/07b/07c)
                     │                                                                                  │
                     ▼                                                                                  ▼
     Stage 3a: operational validation (09b)                                           Stage 3b: operational validation (09a)
                     |                                                                                  |
                     └───────────────────► 09c: shared review gallery for either branch ◄───────────────┘
```

### Stage 1: Event windowing & feature extraction (build the 5 classes)

Run these three independently. Each writes a `catalog_windows`-style CSV that can be loaded and concatenated by every script downstream.

1. **`04a_sta_lta_catalog_windowing.py`:** for every catalog event (earthquake / rockslide / ice quake), precisely windows the   signal, flags whether the P-pick falls inside the window, optionally refines rockslide onsets with a kurtosis picker, computes 7 SNR metrics + 99–103 features. `DETECTION_METHOD` chooses `'groult'` (spectrogram STA/LTA) or `'sta_lta'` (classical).
2. **`04c_regional_event_extraction.py`:** builds the 5th class, *regional* earthquakes (150–1000 km from the massif), windowed around a predicted travel-time arrival instead of a local pick. Same output schema as 04a.
3. **`04d_noise_window_extraction.py`:** builds the 4th class, *local noise*. Real STA/LTA triggers confirmed (by catalog exclusion + cross-station coincidence check) to **not** be a real seismic event. Same output schema as 04a.

Optional rescue step, for classes/stations where too many events fail the SNR quality gate:

4. **`03c_denoiser_event_data.py`:** prepares signal/noise windows and runs DeepDenoiser (Zhu et al. 2019) inference on the low-SNR events.
5. **`03d_rescue_feature_extraction.py`:** recomputes SNR + features on the denoised output, re-applies the quality gate, and writes a rescue catalog in the same schema as 04a (used by 06c).

### Stage 2 — Classification (two independent branches)

Both branches consume the CSVs from Stage 1.

**Branch A — tabular features (Random Forest / HistGradientBoosting):**

6. `06a_train_RF_classifier.py`: baseline Random Forest.
7. `06b_compare_classifiers.py`: benchmarks RF / HGB / KNN / SVM / MLP on the same split.
8. `06c_train_HGB_classifier.py`: final model -> HGB trained on the original + DeepDenoiser-rescued data, with an A/B/C ablation to check whether the rescue gain comes from the denoiser itself or just from more training data. **Produces the `.joblib` model used by `09b`.**

**Branch B — spectrogram images (CNN):**

9. `07a_spectrogram_dataset_build.py`: builds a fixed-size 3-component spectrogram image per sample from the Stage 1 windows.
10. `07b_consolidate_for_colab.py`: packs the many small `.npz` images into a few large archives for upload to Google Drive.
11. `07c_train_cnn_classifier_colab.ipynb`: trains the CNN on Google Colab. **Produces the `.keras` model + normalization stats used by `09a`.**

### Stage 3 — Operational validation on continuous data

Everything above trains/evaluates on curated catalog events. These three scripts are the final check: do the trained models behave sensibly on **raw, continuous, unlabeled data** ? 
Each of 09a/09b runs in two phases (extraction on the cluster, classification wherever the model lives).

12. **`09a_continuous_spectrogram_classification.py`:** Branch B end-to-end test
     - Phase 1 scans continuous days for detections and packs spectrogram images (cluster, no TensorFlow needed)
     - Phase 2 classifies them with the saved `07c` CNN (needs TensorFlow) and writes monthly prediction CSVs + a review gallery
13. **`09b_continuous_tabular_classification.py`:** Branch A end-to-end test
     - Phase 1 extracts features from a continuous scan
     - Phase 2 classifies with the saved `06c` model bundle
14. **`09c_hgb_review_gallery.py`:** visual review of borderline/marginal predictions from either branch (`PIPELINE = "HGB"` or `"CNN"`), to check a class isn't over-triggering.

/!\ There is no ground truth at this stage: the output is for visual/plausibility review, not scoring.

## Calibration / one-time analysis scripts

These were run once to justify a parameter choice that is now hardcoded into the scripts above.
Don't need to re-run them unless recalibrating for a materially different dataset.

- **`03b_feature_selection.py`:** correlation, HGB permutation importance, and subset-size comparison on the 99 features; informed the feature subset used downstream.
- **`04b_method_comparison.py`:** compares 04a's two `DETECTION_METHOD` settings (`'groult'` vs `'sta_lta'`) on the same catalog.
- **`05a_snr_windowing_validation.py`:** checks which SNR metric best predicts whether a detection window is *correctly time-aligned* with the true onset (not a quality gate by itself).
- **`05b_snr_quality_threshold.py`:** what SNR level actually predicts classification usefulness. Together, 05a + 05b fixed the SNR quality gate (`SNR >= 1.70` and `SNR_full_median >= 1.99`) hardcoded into 03c/03d/06b/06c/07a.
- **`05c_kurtosis_onset_comparison.py`:** diagnostic for the kurtosis onset refiner used on rockslides in 04a.

## Report figures

- **`08_report_figures_events.py`:** generates the explanatory figures used in the internship report (not part of the current pipeline)

## Early exploration / not part of the current pipeline

Kept for reference/history. None of these are read by anything downstream.

- **`01_dynamic_station_selection.py`:** early station-coverage exploration.
- **`02a_classical_sta_lta_detection.py`:** early classical STA/LTA demo, superseded by 04a's `DETECTION_METHOD='sta_lta'` option.
- **`02b_spectrogram_sta_lta_detection.py`:** an earlier catalog-less continuous-scan prototype, superseded in practice by 09a/09b's own continuous-scan Phase 1.
- **`03a_feature_extraction.py`:** early feature extractor directly from catalog events (no precise windowing), superseded by 04a.

## How the scripts chain together

Each script writes a timestamped `catalog_windows_<stamp>.csv` (or equivalent) inside its own `results/<script>/run_<timestamp>/` folder, and the *next* script's `CONFIGURATION` section has a path variable (e.g. `CSV_PATH`, `ORIGINAL_CSV`, `RESCUE_CATALOG_CSV`) that you paste that path into by hand before running it.