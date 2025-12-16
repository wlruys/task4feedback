# Generic Hydra-Submitit Runner

A standalone, project-agnostic launcher for Hydra-based experiments. It simplifies running complex experiment sweeps locally or on Slurm clusters using [Submitit](https://github.com/facebookincubator/submitit).

## 🚀 Features

- **Unified Interface**: Switch between local execution and Slurm cluster submission with a single flag.
- **Combinatorial Sweeps**: Easily define parameter sweeps via CLI (`+param=a,b`) or YAML files.
- **Slurm Job Arrays**: Automatically submits sweeps as efficient Slurm Job Arrays (single job ID, easier monitoring).
- **Within-Node Parallelism**: Efficiently utilize powerful cluster nodes by running multiple configurations in parallel within a single Slurm job.
- **Project Agnostic**: Decoupled from your application code. Just provide an entry point and config directory.
- **Hydra Native**: Leverages Hydra's powerful configuration composition.

## 📦 Prerequisites

- Python 3.8+
- `hydra-core`
- `submitit`
- `omegaconf`

```bash
pip install hydra-core submitit omegaconf
```

## 📂 Directory Structure

The runner expects a standard Hydra configuration structure. Here is a recommended layout:

```
project/
├── runner/
│   ├── launcher.py       # The script you run
│   ├── utils.py          # Helper functions
│   └── conf/             # Configuration directory
│       ├── config.yaml   # Main config entry point
│       └── experiment/   # Experiment variations
│           ├── size/
│           │   ├── b4.yaml
│           │   └── b8.yaml
│           └── model/
│               ├── mlp.yaml
│               └── gnn.yaml
└── src/
    └── my_project/
        └── train.py      # Your training script
```

### `runner/conf/config.yaml`
This is the base configuration file. It typically includes defaults and points to your main project configuration.

```yaml
defaults:
  - _self_
  # Load the main config from your project source if needed, 
  # or define base parameters here.
  - /path/to/project/conf/base_config
```

### Experiment Configs (`runner/conf/experiment/...`)
These are small config fragments that override specific parameters. They must use the `# @package _global_` directive to apply changes globally.

```yaml
# runner/conf/experiment/size/b8.yaml
# @package _global_
graph:
  n: 8
```

## 🛠 Usage

### 1. Basic Launch

Run a single configuration locally:

```bash
python runner/launcher.py \
    --entry-point experiments.train:run_training \
    --config-dir runner/conf \
    --local
```

**Important:** The entry point function (e.g., `run_training`) must accept a single argument (the Hydra `DictConfig` object) and should **not** be decorated with `@hydra.main`. The launcher handles the configuration composition and passes the final config object to your function.

### 2. Running on Slurm

Remove the `--local` flag to submit to the cluster. The launcher uses `submitit` to schedule the job.

```bash
python runner/launcher.py \
    --entry-point src.my_project.train:main \
    --config-dir runner/conf \
    --partition learnfair \
    --gpus 1
```

### 3. Parameter Sweeps

You can define sweeps directly on the command line using comma-separated values. The launcher generates the Cartesian product of all options.

```bash
# Generates 4 jobs: (b4, e0.001), (b4, e0.01), (b8, e0.001), (b8, e0.01)
python runner/launcher.py \
    ... \
    +experiment=size/b4,size/b8 \
    +experiment=entropy/e0.001,entropy/e0.01
```

### 4. Sweep Files (YAML)

For complex or reproducible sweeps, define them in a YAML file:

```yaml
# sweeps/my_sweep.yaml
+experiment: 
  - size/b4
  - size/b8
algorithm.lr: [1e-3, 1e-4]
seed: [1, 2, 3]
```

Run with:
```bash
python runner/launcher.py ... --sweep sweeps/my_sweep.yaml
```

### 5. Packs (Single File Definitions)

You can define groups of overrides ("packs") in a single YAML file (default: `runner/packs.yaml`) instead of creating many small config files.

```yaml
# runner/packs.yaml
size:
  b4:
    graph: {n: 4}
  b8:
    graph: {n: 8}
```

Use them just like Hydra config groups:

```bash
python runner/launcher.py ... +size=b4
```

### 6. Within-Node Parallelism (Batching)

To optimize resource usage on clusters, you can group multiple configurations into a single Slurm job and run them in parallel within that job. This is useful for running many small experiments without overwhelming the scheduler or hitting job limits.

**Example:** Run 64 configurations, grouped into 8 Slurm jobs, where each job runs 8 configs in parallel.

```bash
python runner/launcher.py \
    ... \
    --batch-size 8 \
    --jobs 8 \
    --cpus 8
```

*   `--batch-size 8`: Each Slurm job will receive a batch of 8 configurations.
*   `--jobs 8`: The launcher will use 8 parallel workers (processes) within the node to execute the batch.
*   `--cpus 8`: Request 8 CPUs per task from Slurm to ensure each worker has a dedicated CPU.

### 7. Dry Run

Always verify your configurations before submitting hundreds of jobs!

```bash
python runner/launcher.py ... --dry-run
```

## ⚙️ Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--entry-point` | **Required**. Python function to run (`module:function`). Must accept a `DictConfig`. | - |
| `--config-dir` | **Required**. Path to the directory containing `config.yaml`. | - |
| `--config-name` | Base config filename (without extension). | `config` |
| `--sweep` | Path to a YAML file defining a sweep. | `None` |
| `--packs` | Path to a YAML file defining packs. | `runner/packs.yaml` |
| `--name` | Name of the job (appears in Slurm queue). | `config_name_sweep` |
| `--local` | Run jobs sequentially on the local machine. | `False` |
| `--dry-run` | Print generated configurations and exit. | `False` |
| `--jobs`, `-j` | Number of parallel workers (local or within-node). | `1` |
| `--batch-size` | Number of configurations per Slurm job. | `1` |
| `--partition` | Slurm partition to submit to. | `learnfair` |
| `--timeout` | Job timeout in minutes. | `60` |
| `--gpus` | GPUs per node. | `0` |
| `--cpus` | CPUs per task. | `1` |
| `--mem` | Memory per node (e.g., "16GB", "32G"). | `16GB` |
| `--folder` | Directory for Submitit logs (stdout/stderr). | `outputs/submitit_logs` |

## 🎓 Integration Tutorial

To add this runner to a new project:

1.  **Copy Files**: Copy the `runner/` directory (excluding `conf/` if you want to start fresh) to your project root.
2.  **Create Config**: Create `runner/conf/config.yaml`.
3.  **Define Entry Point**: Ensure your training script has a function that accepts a Hydra config.

    ```python
    # src/train.py
    from omegaconf import DictConfig

    def main(cfg: DictConfig):
        print(f"Training with learning rate: {cfg.lr}")

    if __name__ == "__main__":
        # This block is for running train.py directly, not via launcher
        import hydra
        @hydra.main(config_path="../conf", config_name="config")
        def _main(cfg):
            main(cfg)
        _main()
    ```

4.  **Launch**:
    ```bash
    python runner/launcher.py --entry-point src.train:main --config-dir runner/conf --local
    ```

## ❓ Troubleshooting

**Q: My job fails immediately on the cluster but works locally.**
A: Check the Submitit logs in `outputs/submitit_logs/`. Common issues include missing environment variables or paths. Ensure your `entry-point` is importable from the project root.

**Q: How do I pass arguments that aren't in the config?**
A: The launcher only passes the `DictConfig` to your function. Add any necessary parameters to your Hydra config.

**Q: The sweep generates too many jobs.**
A: Use `--dry-run` to check the count. The launcher performs a Cartesian product of all list arguments.
