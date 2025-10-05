rm -rf runner/jacobi_mlp_feature/outputs
rm -rf runner/jacobi_mlp_feature/logs

python runner/expgen.py build \
--yaml runner/jacobi_mlp_feature/experiment.yaml \
--out runner/jacobi_mlp_feature/outputs \
--batch-size 24 --nonstrict

python runner/expgen.py slurm --yaml runner/jacobi_mlp_feature/experiment.yaml --out runner/jacobi_mlp_feature/outputs --launcher '$PWD/runner/run_tmux_launcher.sh' --k-per-session 4 --job-name jacobi_mlp_feature --batch-size 24 --slurm-logs runner/jacobi_mlp_feature/logs --partition skx --time "08:00:00"
