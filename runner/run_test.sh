rm -rf runner/test_launch/outputs
rm -rf runner/test_launch/logs

python runner/expgen.py build \
--yaml runner/test_launch/experiment.yaml \
--out runner/test_launch/outputs \
--batch-size 28 --nonstrict

python runner/expgen.py slurm --yaml runner/test_launch/experiment.yaml --out runner/test_launch/outputs --launcher '$PWD/runner/run_tmux_launcher.sh' --k-per-session 3 --job-name test_job --batch-size 28 --slurm-logs runner/test_launch/logs --partition gg --time "00:10:00"
