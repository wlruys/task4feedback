python3 task_launcher.py --config launch_conf/sweep_256.json --run
python3 generate_problem_size_1_csv.py
python3 task_launcher.py --config launch_conf/sweep.json --run
python3 task_launcher.py --config launch_conf/eval_all.json --run
