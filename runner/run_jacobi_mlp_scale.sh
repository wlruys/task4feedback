rm -rf runner/jacobi_mlp_scale/outputs
rm -rf runner/jacobi_mlp_scale/logs

python runner/expgen.py build \
--yaml runner/jacobi_mlp_scale/experiment.yaml \
--out runner/jacobi_mlp_scale/outputs \
--batch-size 24 --nonstrict


python runner/expgen.py slurm --yaml runner/jacobi_mlp_scale/experiment.yaml --out runner/jacobi_mlp_scale/outputs --launcher '$PWD/runner/run_tmux_launcher.sh' --k-per-session 4 --job-name j_scale --batch-size 24 --slurm-logs runner/jacobi_mlp_scale/logs --partition skx --time "08:00:00"
