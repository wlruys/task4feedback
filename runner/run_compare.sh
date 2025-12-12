rm -rf runner/compare/outputs
rm -rf runner/compare/logs
rm -rf runner/compare/slurm_logs

python runner/expgen.py run --mode local \
  --yaml runner/compare/experiment.yaml \
  --out runner/compare/outputs \
  --batch-size 12 \
  --launcher "$PWD/runner/run_tmux_launcher.sh" \
  --k-per-session 4 \
  --job-name compare \
  --slurm-logs runner/compare/slurm_logs \
