#!/usr/bin/env bash
set -euo pipefail

RUNNER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

EXPERIMENT_DIR="${RUNNER_DIR}/chol_mlp_feature"
YAML_FILE="${EXPERIMENT_DIR}/experiment.yaml"
OUTPUT_DIR="${EXPERIMENT_DIR}/outputs"
SLURM_LOG_DIR="${EXPERIMENT_DIR}/slurm_logs"

BATCH_SIZE=28
K_PER_SESSION=4
JOB_NAME="chol_mlp_feature"
PARTITION="gg"
TIME_LIMIT="08:00:00"
LAUNCHER="${RUNNER_DIR}/run_tmux_launcher.sh"
NONSTRICT=true

rm -rf "${OUTPUT_DIR}" "${SLURM_LOG_DIR}"

NONSTRICT_FLAG=()
if [[ "${NONSTRICT}" == "true" ]]; then
  NONSTRICT_FLAG+=(--nonstrict)
fi

"${PYTHON_BIN}" "${RUNNER_DIR}/expgen.py" build \
  --yaml "${YAML_FILE}" \
  --out "${OUTPUT_DIR}" \
  --batch-size "${BATCH_SIZE}" \
  "${NONSTRICT_FLAG[@]}"

"${PYTHON_BIN}" "${RUNNER_DIR}/expgen.py" slurm \
  --yaml "${YAML_FILE}" \
  --out "${OUTPUT_DIR}" \
  --launcher "${LAUNCHER}" \
  --k-per-session "${K_PER_SESSION}" \
  --job-name "${JOB_NAME}" \
  --batch-size "${BATCH_SIZE}" \
  --slurm-logs "${SLURM_LOG_DIR}" \
  --partition "${PARTITION}" \
  --time "${TIME_LIMIT}"
