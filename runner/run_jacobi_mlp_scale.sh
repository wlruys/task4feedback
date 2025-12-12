rm -rf runner/jacobi_mlp_scale/outputs
rm -rf runner/jacobi_mlp_scale/logs

python runner/expgen.py run --mode slurm \
  --yaml runner/jacobi_mlp_scale/experiment.yaml \
  --out runner/jacobi_mlp_scale/outputs \
  --batch-size 24 \
  --launcher '$PWD/runner/run_tmux_launcher.sh' \
  --k-per-session 4 \
  --job-name j_scale \
  --slurm-logs runner/jacobi_mlp_scale/logs \
  --partition skx \
  --time "08:00:00"
