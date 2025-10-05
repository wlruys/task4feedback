rm -rf runner/cholesky_gnn_feature/outputs
rm -rf runner/cholesky_gnn_feature/logs
rm -rf runner/cholesky_gnn_feature/slurm_logs

python runner/expgen.py build \
--yaml runner/jacobi_gnn_feature/experiment.yaml \
--out runner/cholesky_gnn_feature/outputs \
--batch-size 12 --nonstrict

python runner/expgen.py slurm --yaml runner/cholesky_gnn_feature/experiment.yaml --out runner/cholesky_gnn_feature/outputs --launcher '$PWD/runner/run_tmux_launcher.sh' --k-per-session 4 --job-name cholesky_gnn_feature --batch-size 12 --slurm-logs runner/cholesky_gnn_feature/slurm_logs --partition skx --time "12:00:00"

