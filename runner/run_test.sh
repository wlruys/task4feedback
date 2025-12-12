rm -rf runner/test_launch/outputs
rm -rf runner/test_launch/logs

python runner/expgen.py run --mode slurm \
  --yaml runner/test_launch/experiment.yaml \
  --out runner/test_launch/outputs \
  --batch-size 24 \
  --launcher "$PWD/runner/run_tmux_launcher.sh" \
  --k-per-session 3 \
  --job-name test_job \
  --slurm-logs runner/test_launch/logs \
  --partition skx-dev \
  --time "00:10:00"
