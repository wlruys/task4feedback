rm -rf runner/jacobi_mlp_scale/outputs
rm -rf runner/jacobi_mlp_scale/logs

python runner/expgen.py build \
--yaml runner/jacobi_mlp_scale/experiment.yaml \
--out runner/jacobi_mlp_scale/outputs \
--batch-size 24 --nonstrict

python runner/expgen.py local \
  --yaml runner/jacobi_mlp_scale/experiment.yaml \
  --out runner/jacobi_mlp_scale/outputs \
  --batch-size 4 \
  --launcher "$PWD/runner/run_tmux_launcher.sh" \
  --k-per-session 3 \
  --job-name mylocal \
  --log-dir runner/jacobi_mlp_scale/logs/
