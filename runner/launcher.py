import sys
import os
import argparse
import logging
import submitit
import traceback
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
from omegaconf import OmegaConf
from hydra import compose, initialize_config_dir
from concurrent.futures import ProcessPoolExecutor, as_completed

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.logging import RichHandler
from rich.traceback import install
from rich import box

from runner.utils import parse_overrides, load_sweep_file, import_entry_point, load_packs, expand_pack_overrides

# Install rich traceback handler for nicer error messages
install(show_locals=False)

console = Console()
log = logging.getLogger("launcher")

def setup_logging():
    logging.basicConfig(
        level="INFO",
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)]
    )

class BatchExecutor:
    """
    Executes a list of configurations in parallel within a single job.
    Handles structured logging for each configuration, ensuring outputs
    are saved to specific directories.
    """
    def __init__(self, job_func, max_workers=1, log_dir=None):
        self.job_func = job_func
        self.max_workers = max_workers
        self.log_dir = Path(log_dir) if log_dir else None
    
    def _run_one(self, cfg, idx, job_id, task_id):
        # If no log_dir is provided, just run the function directly
        if not self.log_dir:
            self.job_func(cfg)
            return

        # Structure: log_dir / job_id / task_id / config_idx
        # task_id is the Slurm array index (or 0)
        # config_idx is the index within the batch
        output_dir = self.log_dir / str(job_id) / str(task_id) / str(idx)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        stdout_file = output_dir / "stdout.log"
        stderr_file = output_dir / "stderr.log"
        config_file = output_dir / "config.yaml"
        
        # Save the specific configuration for this run
        with open(config_file, "w") as f:
            OmegaConf.save(cfg, f)
        
        # Run with stdout/stderr redirection
        with open(stdout_file, "w") as f_out, open(stderr_file, "w") as f_err:
            with redirect_stdout(f_out), redirect_stderr(f_err):
                print(f"Job ID: {job_id}")
                print(f"Task ID: {task_id}")
                print(f"Batch Index: {idx}")
                print(f"Output Directory: {output_dir}")
                print("-" * 40)
                try:
                    self.job_func(cfg)
                except Exception as e:
                    print(f"\nEXCEPTION OCCURRED: {e}", file=sys.stderr)
                    traceback.print_exc(file=sys.stderr)
                    raise e

    def __call__(self, configs):
        # Determine execution context (Slurm or Local)
        try:
            env = submitit.JobEnvironment()
            job_id = env.job_id
            task_id = env.global_rank
        except RuntimeError:
            job_id = "local"
            task_id = 0

        if self.max_workers <= 1:
            for i, cfg in enumerate(configs):
                self._run_one(cfg, i, job_id, task_id)
        else:
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [
                    executor.submit(self._run_one, cfg, i, job_id, task_id) 
                    for i, cfg in enumerate(configs)
                ]
                # Wait for all futures to complete to propagate exceptions if any
                for f in futures:
                    f.result()

def get_parser():
    parser = argparse.ArgumentParser(description="Generic Hydra-Submitit Launcher")
    
    # Job Configuration
    parser.add_argument("--entry-point", required=True, help="Python entry point (module:function)")
    parser.add_argument("--config-dir", required=True, help="Path to Hydra config directory")
    parser.add_argument("--config-name", default="config", help="Base config name")
    parser.add_argument("--sweep", help="Path to sweep YAML file")
    parser.add_argument("--packs", default="runner/packs.yaml", help="Path to packs definition file")
    parser.add_argument("--name", help="Job name")
    
    # Execution Mode
    parser.add_argument("--dry-run", action="store_true", help="Print configurations without running")
    parser.add_argument("--local", action="store_true", help="Run locally (no submitit)")
    parser.add_argument("--jobs", "-j", type=int, default=1, help="Number of parallel jobs (local or per Slurm task)")
    parser.add_argument("--batch-size", type=int, default=1, help="Number of configs per Slurm task")
    
    # Submitit Configuration
    parser.add_argument("--partition", default="learnfair", help="SLURM partition")
    parser.add_argument("--timeout", type=int, default=60, help="Timeout in minutes")
    parser.add_argument("--folder", default="outputs/submitit_logs", help="Submitit log folder")
    parser.add_argument("--gpus", type=int, default=0, help="GPUs per node")
    parser.add_argument("--cpus", type=int, default=1, help="CPUs per task")
    parser.add_argument("--mem", default="16GB", help="Memory per node")
    
    return parser

def generate_configs(args, hydra_overrides):
    # 1. Load Sweeps
    if args.sweep:
        log.info(f"Loading sweep from [bold cyan]{args.sweep}[/bold cyan]")
        try:
            sweep_args = load_sweep_file(args.sweep)
            hydra_overrides.extend(sweep_args)
        except Exception as e:
            log.error(f"Error loading sweep file: {e}")
            sys.exit(1)

    # 2. Parse Overrides
    overrides_list = parse_overrides(hydra_overrides)
    
    # 3. Load Packs
    packs = load_packs(args.packs)
    if packs:
        log.info(f"Loaded packs from [bold cyan]{args.packs}[/bold cyan]")
    
    configs = []
    abs_config_dir = Path(args.config_dir).resolve()
    
    if not abs_config_dir.exists():
        log.error(f"Config directory not found: {abs_config_dir}")
        sys.exit(1)

    with console.status(f"[bold green]Generating {len(overrides_list)} configurations...[/bold green]"):
        with initialize_config_dir(config_dir=str(abs_config_dir), version_base=None):
            for overrides in overrides_list:
                expanded_overrides = expand_pack_overrides(overrides, packs)
                try:
                    cfg = compose(config_name=args.config_name, overrides=expanded_overrides)
                    OmegaConf.resolve(cfg)
                    configs.append(cfg)
                except Exception as e:
                    log.error(f"Error composing config with overrides {expanded_overrides}: {e}")
                    sys.exit(1)
                
    return configs, overrides_list

def run_local(job_func, configs, jobs=1):
    console.print()
    console.print(Panel(f"[bold]Running {len(configs)} jobs locally[/bold]\nWorkers: {jobs}", title="Local Execution", border_style="green"))
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Processing...", total=len(configs))
        
        if jobs > 1:
            with ProcessPoolExecutor(max_workers=jobs) as executor:
                future_to_idx = {executor.submit(job_func, cfg): i for i, cfg in enumerate(configs)}
                for future in as_completed(future_to_idx):
                    i = future_to_idx[future]
                    try:
                        future.result()
                    except Exception as e:
                        log.error(f"Job {i+1} failed: {e}")
                    progress.advance(task)
        else:
            for i, cfg in enumerate(configs):
                try:
                    job_func(cfg)
                except Exception as e:
                    log.error(f"Job {i+1} failed: {e}")
                progress.advance(task)
    
    console.print("[bold green]All local jobs completed.[/bold green]")

def run_slurm(job_func, configs, args):
    executor = submitit.AutoExecutor(folder=args.folder)
    
    try:
        mem_gb = int(str(args.mem).upper().replace("GB", "").replace("G", ""))
    except ValueError:
        mem_gb = 16

    executor.update_parameters(
        timeout_min=args.timeout,
        slurm_partition=args.partition,
        gpus_per_node=args.gpus,
        tasks_per_node=1,
        cpus_per_task=args.cpus,
        mem_gb=mem_gb,
        name=args.name if args.name else f"{args.config_name}_sweep"
    )
    
    with console.status("[bold yellow]Submitting to Slurm...[/bold yellow]"):
        try:
            # Always use batching logic to ensure consistent logging structure
            # If batch_size is 1, we still use BatchExecutor to get the per-job logging
            batch_size = args.batch_size if args.batch_size > 0 else 1
            chunks = [configs[i:i + batch_size] for i in range(0, len(configs), batch_size)]
            
            # Pass log folder to executor for structured logging
            batch_executor = BatchExecutor(job_func, max_workers=args.jobs, log_dir=args.folder)
            
            jobs = executor.map_array(batch_executor, chunks)
            
            num_slurm_jobs = len(chunks)
            mode = f"Batched (Size: {batch_size})" if batch_size > 1 else "Individual"
            
        except Exception as e:
            log.error(f"Submission failed: {e}")
            return

    # Summary Table
    grid = Table.grid(expand=True)
    grid.add_column()
    grid.add_column(justify="right")
    
    grid.add_row("Partition", args.partition)
    grid.add_row("Total Configs", str(len(configs)))
    grid.add_row("Slurm Jobs", str(num_slurm_jobs))
    grid.add_row("Mode", mode)
    grid.add_row("Within-Node Workers", str(args.jobs))
    grid.add_row("CPUs/Task", str(args.cpus))
    grid.add_row("GPUs/Node", str(args.gpus))
    grid.add_row("Log Folder", args.folder)
    if jobs:
        grid.add_row("Job ID", str(jobs[0].job_id))
        grid.add_row("Output Structure", f"{args.folder}/<job_id>/<task_id>/<config_idx>/")

    console.print(Panel(grid, title="[bold blue]Slurm Submission Successful[/bold blue]", border_style="green"))

def main():
    setup_logging()
    parser = get_parser()
    args, hydra_overrides = parser.parse_known_args()
    
    try:
        job_func = import_entry_point(args.entry_point)
    except Exception as e:
        log.critical(f"Error loading entry point: {e}")
        sys.exit(1)

    configs, overrides_list = generate_configs(args, hydra_overrides)

    if not configs:
        log.warning("No configurations generated.")
        return

    if args.dry_run:
        table = Table(title="Generated Configurations", box=box.ROUNDED)
        table.add_column("ID", justify="right", style="cyan", no_wrap=True)
        table.add_column("Overrides", style="magenta")
        
        for i, overrides in enumerate(overrides_list):
            # Format overrides nicely
            formatted_overrides = "\n".join(overrides) if len(overrides) > 3 else ", ".join(overrides)
            table.add_row(str(i+1), formatted_overrides)
            
        console.print(table)
        console.print(f"\n[bold]Total configurations:[/bold] {len(configs)}")
        return

    if args.local:
        run_local(job_func, configs, args.jobs)
    else:
        run_slurm(job_func, configs, args)

if __name__ == "__main__":
    main()
