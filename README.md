# Task4Feedback
Time granularity is micro seconds.
Data size granularity is bytes.

Time to move a data is calculated by
```
{Size of data} / {Bandwidth} * 1e6
```

### Finding memory size

For size sweep, change default config in conf/size_sweep.yaml
`mpirun -n 4 python3 size_sweep.py`

Go to results and check the memory where it is interesting.
Also note which level_chunk of Oracle showed the fastest speed up.
In const_vs_inf.py file, set start to 10~20GB smaller than the interesting point.
`python3 const_vs_inf.py --config-name 8x8x128_dynamic_corners_cnn`

### Generating evaluation states

Use the Oracle chunk and level size to pickle eval.
`python3 pickle_eval.py --config-name 8x8x128_dynamic_corners_cnn`

Move pickled file to pickled_evaluation folder.
Don't forget to change base yaml file to point to the pickle file.

`python3 do_rollout.py --config-name 8x8x128_dynamic_corners_cnn reward.verbose=True reward.gamma=1 reward.uniform_reward_scale=10`



```
core = 0
step = 4
id = 0
prj_name = "8x8x128_corners"
device_load = True
for i in range(1):
    for gamma in [0.99, 1, 0.9]:
        for scale in [0.1, 1, 10, 100]:
            for ent in [0.00025, 0.001, 0]:
                for v in ["A", "C", "D", "G"]:
                    if core < 72 and core + step > 72:
                        core = 72
                    print(
                        f"taskset -c {core}-{core+step-1} python3 train.py --config-name 8x8x128_dynamic_corners_cnn reward.verbose=False feature.add_device_load={device_load} feature.observer.version={v} wandb.project={prj_name} algorithm.ent_coef={ent} reward.gamma={gamma} reward.uniform_reward_scale={scale} wandb.name={gamma}_{scale}_{ent}_{v} > /dev/null 2>&1 & sleep {30 if core==0 else 10}s"
                    )

                    core += step
                    id += 1
                    if core >= 120:
                        core = 0
                        print()
```

`python3 run_model.py --config-name 8x8x128_dynamic_corners_cnn feature.observer.version=D`



`python3 run_single_graph.py --config-name 8x8x128_dynamic_corners_cnn `

### Notes
```
if cfg.feature.observer.version in "DFGH":
                    norm.loc[-4:] = 0.0
                    norm.scale[-4:] = 1.0
```
is applied at helper/env.py to preserve one-hot encoding