rm -rf runner/jacobi_gnn_feature_2/outputs
rm -rf runner/jacobi_gnn_feature_2/logs
rm -rf runner/jacobi_gnn_feature_2/slurm_logs

python runner/expgen.py run --mode slurm \
  --yaml runner/jacobi_gnn_feature/experiment.yaml \
  --out runner/jacobi_gnn_feature_2/outputs \
  --batch-size 12 \
  --launcher '$PWD/runner/run_tmux_launcher.sh' \
  --k-per-session 4 \
  --job-name jacobi_gnn_feature_ipdps \
  --slurm-logs runner/jacobi_gnn_feature_2/slurm_logs \
  --partition skx \
  --time "12:00:00"
