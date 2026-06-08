#!/bin/bash
#OAR -n 03c_train_deepdenoiser
#OAR -l /nodes=1/core=4,walltime=24:00:00

# Modules loading
source /soft/env.bash
module load python/python3.11

# Activate the seismo virtualenv (contains tensorflow, numpy, scipy)
source /data/failles/louisels/envs/seismo/bin/activate

# ── Paths (update if you re-ran 03c and got a new run folder) ─────────────────
DATA_DIR=/data/failles/louisels/project/results/outputs_03c/run_20260605_102106
DEEPDENOISER_DIR=/data/failles/louisels/project/src/deepdenoiser

# ── Launch training ───────────────────────────────────────────────────────────
cd ${DEEPDENOISER_DIR}

python3.11 train.py \
  --train_signal_dir ${DATA_DIR}/signal/ \
  --train_signal_list ${DATA_DIR}/signal_list.csv \
  --train_noise_dir  ${DATA_DIR}/noise/ \
  --train_noise_list ${DATA_DIR}/noise_list.csv \
  --log_dir          ${DATA_DIR}/log \
  --epochs           50 \
  --batch_size       20 \
  --learning_rate    0.001 \
  --summary          True

# ── After training ────────────────────────────────────────────────────────────
# The trained model checkpoint will be saved in:
#   ${DATA_DIR}/log/<YYMMDD-HHMMSS>/model_<epoch>.ckpt.*
#
# Once training is done, set MODEL_DIR in 03c_denoiser_icequake_data.py to that
# checkpoint folder and re-run 03c to denoise the rescue targets:
#   MODEL_DIR = "${DATA_DIR}/log/<YYMMDD-HHMMSS>"
