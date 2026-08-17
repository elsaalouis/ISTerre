"""
08c_report_figures_cnn.py
==========================
ISTerre internship — Environmental seismology in glaciology
Author : Elsa Louis
Date   : August 2026

Goal
----
Report figures illustrating the CNN spectrogram classifier's results, as a
separate script from the 07 family on purpose:
  07a_spectrogram_dataset_build.py  builds the dataset (cluster: SDS + FDSN)
  07b_train_cnn_classifier_colab.ipynb  trains the model (Colab: needs a GPU)
  08c (this script)                 only READS what 07a/07b already produced
                                     and turns it into report figures
-> no cluster, no SDS/FDSN, no GPU needed here — everything it needs
   (packed spectrogram images + the trained .keras model + its normalization
   stats) is local once you've downloaded 07b's Colab log folder. Run this
   directly on your machine (the glacier-seismo conda env already has
   TensorFlow per environment.yml).

Two figure types, both TEST-SPLIT examples, one PNG per example (not a grid):
  1. Training input gallery — a (n_freq, n_time, 3) [Z, N, E] dB spectrogram
     rendered as a single R=Z/G=N/B=E composite image: "the object the CNN
     actually sees" as one picture, instead of 3 separate grayscale panels.
  2. Grad-CAM gallery — for CORRECTLY classified examples, which
     time-frequency region drove the prediction (same recipe as 07b's Cell
     14: gradient of the predicted class w.r.t. the last Conv2D layer's
     activations), generalized from 07b's 1-example-per-class preview to
     N_GRADCAM_EXAMPLES_PER_CLASS.

IMPORTANT — MODEL_DIR and DATA_DIR must be a MATCHED pair (same class list,
same train/val/test split) or predictions/labels won't line up. Check
MODEL_DIR's run_log_*.txt for its "CLASS_NAMES = [...]" and manifest row
counts, and set CLASS_NAMES below to match EXACTLY (same order — it's a
plain index lookup, not inferred from the data).

Output
------
  examples/spec/<abbr>/fig_spec_<class>_NN_<net>_<sta>_<stamp>.png
  examples/gradcam/<abbr>/fig_gradcam_<class>_NN_<net>_<sta>_<stamp>.png
  run.log
"""



# =============================================================================
# SECTION 1 — CONFIGURATION
# =============================================================================

# -- Input: a 07a run (packed test-split spectrograms + shared axes) ----------
# Must match MODEL_DIR below (same dataset build -- see CLASS_NAMES note above).
DATA_DIR  = r"C:\Users\elsa.louis\OneDrive - ESTIA\Documents\4 ISTERRE\project\results\07a_spectrogram_dataset_build\65-20-15_5classes_20260806_154405"
SPEC_TEST_PATH = DATA_DIR + r"\spectrograms_test.npz"
FREQ_AXIS_PATH = DATA_DIR + r"\freq_axis.npy"
TIME_AXIS_PATH = DATA_DIR + r"\time_axis.npy"

# -- Input: a 07b Colab run (trained model + normalization stats) -------------
# Downloaded from Google Drive's colab_cnn_training_spectrogram/log/ folder.
MODEL_DIR  = r"C:\Users\elsa.louis\OneDrive - ESTIA\Documents\4 ISTERRE\project\results\07b_cnn_classifier\5classes_20260812_130248"
MODEL_PATH = MODEL_DIR + r"\best_model.keras"   # or best_model.keras
NORM_STATS_PATH = MODEL_DIR + r"\normalization_stats.npz"

# -- Output ---------------------------------------------------------------------
OUTPUT_DIR = r"C:\Users\elsa.louis\OneDrive - ESTIA\Documents\4 ISTERRE\project\results\08c_report_figures_cnn"

# -- MUST match MODEL_DIR's training run EXACTLY (same order -- this is a plain
#    index lookup: model output index i <-> CLASS_NAMES[i]). Cross-check
#    against "CLASS_NAMES = [...]" in MODEL_DIR's run_log_*.txt. -------------
CLASS_NAMES = ["earthquake", "regional", "rockslide", "ice quake", "noise"]
CLASS_ABBR  = {"earthquake": "eq", "rockslide": "rs", "ice quake": "iq",
               "noise": "no", "regional": "re"}

# -- How many examples per class, one PNG each ---------------------------------
N_SPEC_EXAMPLES_PER_CLASS    = 30    # training-input gallery (Z/N/E composite)
N_GRADCAM_EXAMPLES_PER_CLASS = 5    # Grad-CAM gallery (correctly-classified only,
                                    # same convention as 07b Cell 14)
EXAMPLE_SEED = 42

# -- Display-only RGB scaling for the spectrogram gallery (see
#    plot_spectrogram_rgb_example's docstring in visualization.py) -----------
RGB_PCTL_LO, RGB_PCTL_HI = 1.0, 99.0



# =============================================================================
# SECTION 2 — SETUP
# =============================================================================

import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np

import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_setup import create_run_dir, setup_logging, set_matplotlib_defaults
from visualization import plot_spectrogram_rgb_example, plot_gradcam_example

RUN_DIR, STAMP = create_run_dir(OUTPUT_DIR)
log_file, log_path = setup_logging(
    RUN_DIR, "08c_report_figures_cnn.py",
    extra_info=(f"DATA_DIR: {DATA_DIR}\nMODEL_DIR: {MODEL_DIR}\n"
                f"CLASS_NAMES: {CLASS_NAMES}")
)
set_matplotlib_defaults()

print(f"  Python executable : {sys.executable}")
print(f"  NumPy version     : {np.__version__}")

try:
    import tensorflow as tf
except ImportError:
    print("\n[ERROR] TensorFlow is not installed in this environment.")
    print("        This script needs it to load the .keras model and compute Grad-CAM")
    print("        (unlike 03d/08a/08b, which are TensorFlow-free).")
    print("        pip install tensorflow   (or: conda activate glacier-seismo)")
    log_file.close()
    sys.exit(1)

print(f"  TensorFlow version: {tf.__version__}")



# =============================================================================
# SECTION 3 — LOAD DATA (test split only, matches 07b Cell 14's convention)
# =============================================================================

print(f"\n{'='*65}")
print("  STEP 1 — Loading test-split spectrograms")
print(f"{'='*65}")

for label, path in [("SPEC_TEST_PATH", SPEC_TEST_PATH), ("FREQ_AXIS_PATH", FREQ_AXIS_PATH),
                    ("TIME_AXIS_PATH", TIME_AXIS_PATH), ("MODEL_PATH", MODEL_PATH),
                    ("NORM_STATS_PATH", NORM_STATS_PATH)]:
    if not os.path.isfile(path):
        print(f"[ERROR] {label} not found: {path}")
        log_file.close()
        sys.exit(1)

freq_axis = np.load(FREQ_AXIS_PATH)
time_axis = np.load(TIME_AXIS_PATH)
print(f"  Frequency axis: {len(freq_axis)} bins, 0-{freq_axis.max():.1f} Hz")
print(f"  Time axis     : {len(time_axis)} bins, 0-{time_axis.max():.1f} s")

with np.load(SPEC_TEST_PATH, allow_pickle=False) as d:
    X_test_raw  = d["images"]          # float16, [Z, N, E], as packed by 07a_consolidate_for_colab.py
    y_test_lbl  = d["labels"]
    ev_time     = d["event_time"]
    ev_net      = d["network"]
    ev_sta      = d["station"]

label2idx = {name: i for i, name in enumerate(CLASS_NAMES)}
missing_labels = sorted(set(y_test_lbl.tolist()) - set(CLASS_NAMES))
if missing_labels:
    print(f"[ERROR] Test set contains label(s) not in CLASS_NAMES: {missing_labels}")
    print(f"        CLASS_NAMES={CLASS_NAMES} -- fix Section 1 to match MODEL_DIR's training run.")
    log_file.close()
    sys.exit(1)

y_test = np.array([label2idx[lbl] for lbl in y_test_lbl], dtype="int32")
print(f"  Test set: {X_test_raw.shape[0]:,} samples, shape {X_test_raw.shape[1:]}")
for cls in CLASS_NAMES:
    n = int((y_test_lbl == cls).sum())
    print(f"    {cls:<12s} {n:6,}")



# =============================================================================
# SECTION 4 — LOAD MODEL + NORMALIZATION STATS
# =============================================================================

print(f"\n{'='*65}")
print("  STEP 2 — Loading trained model")
print(f"{'='*65}")

# compile=False -- the saved compile_config references 07b Cell 9's custom
# label-smoothed loss (make_smoothed_sparse_ce's inner loss_fn), a plain
# closure that was never @keras.saving.register_keras_serializable()'d, so
# Keras can't deserialize it back by name outside the notebook it was
# defined in. We only need this model for inference (predict) and for raw
# gradients w.r.t. a conv layer's activations (Grad-CAM, via GradientTape) --
# neither touches the compiled optimizer/loss/metrics, so skipping their
# reconstruction entirely is the correct fix, not a workaround.
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print(f"  Loaded: {MODEL_PATH}")
print(f"  Output classes (model): {model.output_shape[-1]}  |  CLASS_NAMES: {len(CLASS_NAMES)}")
if model.output_shape[-1] != len(CLASS_NAMES):
    print(f"[ERROR] Model has {model.output_shape[-1]} output classes but CLASS_NAMES has "
          f"{len(CLASS_NAMES)} -- these MUST match. Fix Section 1.")
    log_file.close()
    sys.exit(1)

norm_stats    = np.load(NORM_STATS_PATH)
channel_mean  = norm_stats["mean"]
channel_std   = norm_stats["std"]
print(f"  Per-channel mean: {channel_mean.ravel()}")
print(f"  Per-channel std : {channel_std.ravel()}")

X_test_n = (X_test_raw.astype("float32") - channel_mean) / channel_std

print(f"\n{'='*65}")
print("  STEP 3 — Running inference on the test set")
print(f"{'='*65}")

y_proba = model.predict(X_test_n, batch_size=128, verbose=0)
y_pred  = np.argmax(y_proba, axis=1)
acc     = float((y_pred == y_test).mean())
print(f"  Test accuracy: {acc:.3f}  ({(y_pred == y_test).sum():,}/{len(y_test):,})")



# =============================================================================
# SECTION 5 — TRAINING INPUT GALLERY (Z/N/E composite, RAW examples)
# =============================================================================

print(f"\n{'='*65}")
print(f"  STEP 4 — Training input gallery ({N_SPEC_EXAMPLES_PER_CLASS} per class)")
print(f"{'='*65}")

rng = np.random.default_rng(EXAMPLE_SEED)
spec_dir = os.path.join(RUN_DIR, "examples", "spec")

for cls in CLASS_NAMES:
    cls_idx = np.where(y_test_lbl == cls)[0]
    if len(cls_idx) == 0:
        print(f"  [WARN] '{cls}': no test examples available -- skipped.")
        continue
    n_take   = min(N_SPEC_EXAMPLES_PER_CLASS, len(cls_idx))
    selected = rng.choice(cls_idx, size=n_take, replace=False)

    out_dir_cls = os.path.join(spec_dir, CLASS_ABBR.get(cls, cls[:2]))
    os.makedirs(out_dir_cls, exist_ok=True)

    for k, idx in enumerate(selected, start=1):
        image_db = X_test_raw[idx].astype(np.float32)   # raw dB, NOT normalized -- display only
        title_l1 = f"{cls} \u2014 training input example"
        title_l2 = f"{ev_net[idx]}.{ev_sta[idx]} | {str(ev_time[idx])[:19]}"
        out_path = os.path.join(
            out_dir_cls,
            f"fig_spec_{CLASS_ABBR.get(cls, cls[:2])}_{k:02d}_{ev_net[idx]}_{ev_sta[idx]}_{STAMP}.png",
        )
        plot_spectrogram_rgb_example(
            freq_axis, time_axis, image_db,
            title_lines=(title_l1, title_l2), out_path=out_path,
            pctl_lo=RGB_PCTL_LO, pctl_hi=RGB_PCTL_HI,
        )
    print(f"  [OK] '{cls}': {n_take}/{N_SPEC_EXAMPLES_PER_CLASS} examples saved to {out_dir_cls}")



# =============================================================================
# SECTION 6 — GRAD-CAM GALLERY (correctly-classified examples only)
# =============================================================================
# Same recipe as 07b_train_cnn_classifier_colab.ipynb's Cell 14: gradient of
# the predicted class's logit w.r.t. the last Conv2D layer's activations,
# global-average-pooled into per-channel weights, ReLU'd, normalized to [0,1].
# "Correctly classified only" is intentional (matches Cell 14) -- the point is
# to sanity-check what the network keys on when it gets it RIGHT, not to
# diagnose failures.

print(f"\n{'='*65}")
print(f"  STEP 5 — Grad-CAM gallery ({N_GRADCAM_EXAMPLES_PER_CLASS} per class, correct predictions only)")
print(f"{'='*65}")

last_conv_name = None
for layer in model.layers:
    if isinstance(layer, tf.keras.layers.Conv2D):
        last_conv_name = layer.name
if last_conv_name is None:
    print("[ERROR] No Conv2D layer found in the model -- cannot compute Grad-CAM.")
    log_file.close()
    sys.exit(1)
print(f"  Grad-CAM target layer: {last_conv_name}")

grad_model = tf.keras.models.Model(model.inputs, [model.get_layer(last_conv_name).output, model.output])


def compute_gradcam(img_batch, class_idx):
    """img_batch: (1, n_freq, n_time, 3) float32, already normalized.
    Returns a (n_freq, n_time) numpy array, ReLU'd and max-normalized to [0,1]."""
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_batch)
        loss = preds[:, class_idx]
    grads   = tape.gradient(loss, conv_out)
    weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam     = tf.reduce_sum(weights * conv_out, axis=-1)
    cam     = tf.nn.relu(cam)
    cam     = cam / (tf.reduce_max(cam, axis=(1, 2), keepdims=True) + 1e-8)
    return cam.numpy()[0]


gradcam_dir = os.path.join(RUN_DIR, "examples", "gradcam")

for cls in CLASS_NAMES:
    cls_idx_target = label2idx[cls]
    correct_idx = np.where((y_test == cls_idx_target) & (y_pred == cls_idx_target))[0]
    if len(correct_idx) == 0:
        print(f"  [WARN] '{cls}': no correctly-classified test examples -- skipped.")
        continue
    n_take   = min(N_GRADCAM_EXAMPLES_PER_CLASS, len(correct_idx))
    selected = rng.choice(correct_idx, size=n_take, replace=False)

    out_dir_cls = os.path.join(gradcam_dir, CLASS_ABBR.get(cls, cls[:2]))
    os.makedirs(out_dir_cls, exist_ok=True)

    for k, idx in enumerate(selected, start=1):
        img_batch = X_test_n[idx:idx + 1]
        cam = compute_gradcam(img_batch, cls_idx_target)

        confidence = float(y_proba[idx, cls_idx_target])
        z_channel_db = X_test_raw[idx][:, :, 0].astype(np.float32)   # raw dB, Z channel only

        title_l1 = f"{cls} \u2014 Grad-CAM  (P={confidence:.2f})"
        title_l2 = f"{ev_net[idx]}.{ev_sta[idx]} | {str(ev_time[idx])[:19]}"
        out_path = os.path.join(
            out_dir_cls,
            f"fig_gradcam_{CLASS_ABBR.get(cls, cls[:2])}_{k:02d}_{ev_net[idx]}_{ev_sta[idx]}_{STAMP}.png",
        )
        plot_gradcam_example(
            freq_axis, time_axis, z_channel_db, cam,
            title_lines=(title_l1, title_l2), out_path=out_path,
        )
    print(f"  [OK] '{cls}': {n_take}/{N_GRADCAM_EXAMPLES_PER_CLASS} examples saved to {out_dir_cls}")



# =============================================================================
# END
# =============================================================================

from datetime import datetime
print("\n" + "=" * 70)
print(f"  Run finished    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Test accuracy   : {acc:.3f}")
print(f"  All outputs     : {RUN_DIR}")
print(f"  Log file        : {log_path}")
print("=" * 70)
log_file.close()
