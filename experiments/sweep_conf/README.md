# Usage 

To run the sweep, use the following commands:
```bash
wandb sweep sweep_conf/<sweep_config_file>.yaml
wandb agent --count <number to run> <sweep_id>
```

Sweeps can be called in parallel by running multiple agents, on potentially different machines.
This is useful to submit over SLURM. 

