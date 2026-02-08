#!/bin/bash
#SBATCH --job-name=jac_mlp_feature_dynamic_circle
#SBATCH --array=0-5
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=/Users/wlruys/workspace/scheduling/task4feedback/runner/jac_mlp_feature_dynamic_circle/slurm_logs/jac_mlp_feature_dynamic_circle_%A_%a.out
#SBATCH --error=/Users/wlruys/workspace/scheduling/task4feedback/runner/jac_mlp_feature_dynamic_circle/slurm_logs/jac_mlp_feature_dynamic_circle_%A_%a.err
#SBATCH --partition=gg
#SBATCH --account=PHY21005

set -euo pipefail

########## micromamba bootstrap ##########
# Honors MICROMAMBA_EXE or MAMBA_EXE if you’ve set them; falls back to `micromamba` on PATH.
MICROMAMBA="/scratch/06081/wlruys/micromamba/micromamba"
if command -v "$MICROMAMBA" >/dev/null 2>&1; then
  # Initialize the shell integration for bash in a non-interactive context
  eval "$("$MICROMAMBA" shell hook -s bash --root-prefix /scratch/06081/wlruys/micromamba_prefix)"
  # Activate your env; change "py313" to your actual env name if different
  micromamba activate pyt4f
else
  echo "[WARN] micromamba not found (MICROMAMBA_EXE/MAMBA_EXE not set and 'micromamba' not on PATH)." >&2
fi
##########################################

SLURM_JOB_ID="${SLURM_JOB_ID:-nojid}"
SLURM_ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

files=('/Users/wlruys/workspace/scheduling/task4feedback/runner/jac_mlp_feature_dynamic_circle/outputs/batches/batch_000.txt' '/Users/wlruys/workspace/scheduling/task4feedback/runner/jac_mlp_feature_dynamic_circle/outputs/batches/batch_001.txt' '/Users/wlruys/workspace/scheduling/task4feedback/runner/jac_mlp_feature_dynamic_circle/outputs/batches/batch_002.txt' '/Users/wlruys/workspace/scheduling/task4feedback/runner/jac_mlp_feature_dynamic_circle/outputs/batches/batch_003.txt' '/Users/wlruys/workspace/scheduling/task4feedback/runner/jac_mlp_feature_dynamic_circle/outputs/batches/batch_004.txt' '/Users/wlruys/workspace/scheduling/task4feedback/runner/jac_mlp_feature_dynamic_circle/outputs/batches/batch_005.txt')

# Bounds check for the array index
idx="$SLURM_ARRAY_TASK_ID"
if (( idx < 0 || idx >= ${#files[@]} )); then
  echo "[ERROR] SLURM_ARRAY_TASK_ID=$idx is out of range [0, $((${#files[@]}-1))]." >&2
  exit 2
fi

cmdfile="${files[$idx]}"
if [[ ! -f "$cmdfile" ]]; then
  echo "[ERROR] Command file not found: $cmdfile" >&2
  exit 3
fi

echo "[INFO] Job $SLURM_JOB_ID ArrayTask $SLURM_ARRAY_TASK_ID -> file: $cmdfile"

# Use **double quotes** so SLURM variables expand, and namespace logs per job
export TMUX_LOG_DIR="slurm_logs/mylocal/${SLURM_JOB_ID}"
export TMUX_PREFIX="mylocal_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TMUX_LOG_DIR"

# Absolute path to the launcher; call via bash and quote properly
LAUNCHER="/scratch/06081/wlruys/task4feedback/runner/run_tmux_launcher.sh"
if [[ ! -f "$LAUNCHER" ]]; then
  echo "[ERROR] Launcher not found at: $LAUNCHER" >&2
  exit 4
fi

bash "$LAUNCHER" "$cmdfile" 4 "$TMUX_PREFIX" "$TMUX_LOG_DIR"
