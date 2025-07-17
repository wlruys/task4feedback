#!/usr/bin/env python3
import argparse
import subprocess
import sys
import os
import logging
import wandb
import yaml
from pathlib import Path
from typing import Optional


def setup_logging() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    return logging.getLogger(__name__)


def check_env():
    if not os.environ.get("WANDB_API_KEY"):
        print("Error: WANDB_API_KEY environment variable is not set")
        sys.exit(1)


def create_sweep(
    sweep_config_path: str,
    project: Optional[str] = None,
    existing_sweep_id: Optional[str] = None,
    logger: logging.Logger = None,
) -> str:
    try:
        wandb.login()

        if existing_sweep_id:
            logger.info(f"Using existing sweep: {existing_sweep_id}")
            return existing_sweep_id

        # Only validate config file if creating new sweep
        config_file = Path(sweep_config_path)
        if not config_file.exists():
            logger.error(f"Config file not found: {sweep_config_path}")
            sys.exit(1)

        with open(sweep_config_path, "r") as f:
            sweep_config = yaml.safe_load(f)

        sweep_id = wandb.sweep(sweep=sweep_config, project=project)
        logger.info(f"Created sweep: {sweep_id}")
        return sweep_id

    except Exception as e:
        logger.error(f"Failed to create/validate sweep: {e}")
        sys.exit(1)


def submit_slurm_jobs(
    project: str,
    sweep_id: str,
    num_agents: int,
    slurm_script_path: str,
    logger: logging.Logger,
) -> None:
    if not Path(slurm_script_path).exists():
        logger.error(f"SLURM script not found: {slurm_script_path}")
        sys.exit(1)

    try:
        api = wandb.Api()
        entity = api.default_entity
    except Exception as e:
        logger.error(f"Failed to get W&B entity: {e}")
        sys.exit(1)

    failed_count = 0
    logger.info(f"SweepID: {sweep_id}, Project: {project}, Entity: {entity}")

    for i in range(num_agents):
        agent_num = i + 1
        logger.info(f"Submitting agent {agent_num}/{num_agents}")

        try:
            cmd = ["sbatch", slurm_script_path, sweep_id, project, entity]
            result = subprocess.run(
                cmd, check=True, capture_output=True, text=True, timeout=60
            )

            if result.stdout:
                logger.info(f"Agent {agent_num}: {result.stdout.strip()}")

        except subprocess.TimeoutExpired:
            logger.error(f"Agent {agent_num} submission timed out")
            failed_count += 1
        except subprocess.CalledProcessError as e:
            logger.error(
                f"Agent {agent_num} failed: {e.stderr.strip() if e.stderr else 'Unknown error'}"
            )
            failed_count += 1
        except Exception as e:
            logger.error(f"Agent {agent_num} error: {e}")
            failed_count += 1

    if failed_count > 0:
        logger.warning(f"{failed_count}/{num_agents} agents failed to submit")
        if failed_count == num_agents:
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Create W&B sweep and launch parallel agents via SLURM"
    )
    parser.add_argument("--config", help="Path to sweep YAML config file")
    parser.add_argument("--project", help="W&B project name", default="example")
    parser.add_argument(
        "--agents", type=int, default=1, help="Number of parallel agents (default: 1)"
    )
    parser.add_argument(
        "--slurm-script",
        default="sweep_conf/submit.sh",
        help="Path to SLURM script",
    )
    parser.add_argument(
        "--sweep_id", help="Existing sweep ID to use instead of creating new one"
    )

    args = parser.parse_args()

    if args.agents < 1:
        print("Error: Number of agents must be at least 1")
        sys.exit(1)

    logger = setup_logging()

    try:
        check_env()
        sweep_id = create_sweep(args.config, args.project, args.sweep_id, logger)
        submit_slurm_jobs(
            args.project, sweep_id, args.agents, args.slurm_script, logger
        )
        logger.info("Completed successfully")

    except KeyboardInterrupt:
        logger.info("Cancelled by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
