rm -rf runner/jacobi_mlp_feature/outputs
rm -rf runner/jacobi_mlp_feature/logs

python runner/expgen.py run --mode slurm \
  --yaml runner/jacobi_mlp_feature/experiment.yaml \
  --out runner/jacobi_mlp_feature/outputs \
  --batch-size 24 \
  --launcher "$PWD/runner/run_tmux_launcher.sh" \
  --k-per-session 4 \
  --job-name jacobi_mlp_feature \
  --slurm-logs runner/jacobi_mlp_feature/logs \
  --partition spr \
  --time "08:00:00"
