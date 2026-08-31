"""
denoiser_utils.py
==================
ISTerre internship 
Author : Elsa Louis
Date   : July 2026

Shared helpers for invoking DeepDenoiser's predict.py (Zhu et al., 2019) from any
pipeline script that needs to run inference on a folder of .npz windows —
currently used by 03c_denoiser_event_data.py (rescue targets) and
03e_denoiser_good_signal_test.py (already-good-SNR sample, for comparison).

Functions
---------
  fix_checkpoint_paths(model_dir)                          — rewrite TF checkpoint to relative paths
  validate_npz_files(data_dir, csv_path)                    — pre-flight .npz load check
  run_deepdenoiser_predict(...)                             — validate + fix + invoke predict.py
"""

import os
import re
import sys
import site
import subprocess
import traceback

import numpy as np
import pandas as pd


def fix_checkpoint_paths(model_dir):
    """
    Rewrite the TF 'checkpoint' file in model_dir so model_checkpoint_path / all_model_checkpoint_paths use just the 
    basename instead of the absolute path recorded at save time
    """
    ckpt_path = os.path.join(model_dir, "checkpoint")
    if not os.path.isfile(ckpt_path):
        return

    with open(ckpt_path, "r") as f:
        lines = f.readlines()

    fixed_lines = []
    changed = False
    for line in lines:
        m = re.match(r'^(model_checkpoint_path|all_model_checkpoint_paths):\s*"(.*)"\s*$', line)
        if m:
            key, path = m.group(1), m.group(2)
            base = os.path.basename(path)
            if base != path:
                changed = True
            fixed_lines.append(f'{key}: "{base}"\n')
        else:
            fixed_lines.append(line)

    if changed:
        with open(ckpt_path, "w") as f:
            f.writelines(fixed_lines)
        print(f"  [FIXED] Rewrote checkpoint to relative paths: {ckpt_path}")


def validate_npz_files(data_dir, csv_path):
    """
    Pre-flight check: try to np.load() every file listed in csv_path (must have a 'fname' column) before handing the list to predict.py

    Parameters
    ----------
    data_dir : str — folder containing the .npz files
    csv_path : str — CSV with a 'fname' column (rescue_list.csv / goodtest_list.csv format)

    Returns
    -------
    valid_csv_path : str or None
    n_total, n_bad : int, int
    """
    df  = pd.read_csv(csv_path)
    bad = []
    for fname in df['fname']:
        fpath = os.path.join(data_dir, fname)
        try:
            with np.load(fpath) as npz:
                _ = npz['data'].shape   # force a real read, not just the zip header
        except Exception as e:
            bad.append((fname, str(e)))

    n_total = len(df)
    n_bad   = len(bad)

    if n_bad == 0:
        print(f"  [OK] All {n_total} files in {os.path.basename(csv_path)} loaded fine.")
        return csv_path, n_total, 0

    print(f"  [WARN] {n_bad}/{n_total} files failed to load — skipping them:")
    for fname, err in bad[:10]:
        print(f"           {fname}: {err}")
    if n_bad > 10:
        print(f"           ... and {n_bad - 10} more")

    df_ok = df[~df['fname'].isin({b[0] for b in bad})]
    if len(df_ok) == 0:
        return None, n_total, n_bad

    valid_csv_path = csv_path[:-4] + "_valid.csv"
    df_ok.to_csv(valid_csv_path, index=False)
    print(f"  [SAVED] {valid_csv_path}  ({len(df_ok)} good files)")
    return valid_csv_path, n_total, n_bad


def run_deepdenoiser_predict(data_dir, csv_path, model_dir, output_dir,
                              deepdenoiser_dir, sampling_rate=100):
    """
    Validate + fix checkpoint + run DeepDenoiser's predict.py on a folder of .npz windows, in inference mode (--save_signal)

    Parameters
    ----------
    data_dir         : str — folder containing the .npz files to denoise
    csv_path         : str — CSV with a 'fname' column listing which files to process
    model_dir        : str — trained DeepDenoiser checkpoint folder
    output_dir       : str — where predict.py should write its output (results/ subfolder)
    deepdenoiser_dir : str — folder containing predict.py
    sampling_rate    : int — passed through to predict.py (default 100 Hz)

    Returns
    -------
    ok         : bool — True if predict.py ran and exited 0
    results_dir: str or None — output_dir/results (predict.py's actual output location) if ok, else None
    """
    predict_script = os.path.join(deepdenoiser_dir, "predict.py")
    os.makedirs(output_dir, exist_ok=True)

    n_rows = len(pd.read_csv(csv_path))
    if n_rows == 0:
        print("  [WARN] Input list is empty — nothing to denoise.")
        return False, None

    if not os.path.isfile(predict_script):
        print(f"  [ERROR] predict.py not found at {predict_script}")
        return False, None

    fix_checkpoint_paths(model_dir)

    print("\n  Validating .npz files before running predict.py ...")
    valid_csv_path, n_checked, n_bad = validate_npz_files(data_dir, csv_path)

    if valid_csv_path is None:
        print("  [ERROR] Every file failed to load — nothing to denoise. "
              "See the [WARN] lines above (likely a OneDrive sync issue, or a "
              "data-format mismatch if the error text is identical across all files).")
        return False, None

    cmd = [
        sys.executable, predict_script,
        "--format",        "numpy",
        "--data_dir",      data_dir,
        "--data_list",     valid_csv_path,
        "--model_dir",     model_dir,
        "--output_dir",    output_dir,
        "--save_signal",
        "--sampling_rate", str(sampling_rate),
    ]

    env = os.environ.copy()
    try:
        env_site_packages = [p for p in site.getsitepackages() if os.path.isdir(p)]
    except Exception:
        env_site_packages = []
    if env_site_packages:
        old_pp = env.get("PYTHONPATH", "")
        new_pp = os.pathsep.join(env_site_packages + ([old_pp] if old_pp else []))
        env["PYTHONPATH"] = new_pp
        print(f"  [INFO] Prepending env site-packages to PYTHONPATH so predict.py's "
              f"tensorflow/protobuf resolve from the seismo env, not an inherited "
              f"system path: {env_site_packages}")

    print(f"  Command : {' '.join(cmd)}\n")
    try:
        subprocess.run(cmd, check=True, env=env)
        results_dir = os.path.join(output_dir, "results")
        print(f"\n  [OK] DeepDenoiser inference completed.")
        print(f"       Denoised files → {results_dir}/")
        return True, results_dir
    except subprocess.CalledProcessError as e:
        print(f"\n  [ERROR] predict.py exited with code {e.returncode}")
        traceback.print_exc()
        return False, None
