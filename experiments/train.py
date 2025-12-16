
import hydra
import wandb
import os
from omegaconf import DictConfig, OmegaConf, open_dict
from task4feedback.exp_utils.training import (
    build_graph_builder,
    initialize_env_and_model,
    load_normalization,
    maybe_log_model_to_wandb,
    parameter_counts,
    persist_config_and_normalization,
    prepare_logging_config,
    prepare_training_components,
    set_global_seeds,
)

from task4feedback.interface.wrappers import *
from task4feedback.ml.models import *

# torch.multiprocessing.set_sharing_strategy("file_descriptor")
# torch.multiprocessing.set_sharing_strategy("file_system")

from pathlib import Path
from hydra.core.hydra_config import HydraConfig


from task4feedback.ml.algorithms import Trainer, get_algorithm, AlgorithmConfig

def configure_training(cfg: DictConfig):
    logging_config, _best_policy_path, _model_ctx = prepare_logging_config(cfg)

    normalization, _norm_ctx = load_normalization(cfg)
    graph_builder = build_graph_builder(cfg)
    _, normalization, env_fn, model_bundle = initialize_env_and_model(cfg, graph_builder, normalization)

    alg_config, optimizer, lr_scheduler, eval_config, eval_location = prepare_training_components(cfg)
    persist_config_and_normalization(cfg, normalization)

    total_params, trainable_params = parameter_counts(model_bundle.model)
    maybe_log_model_to_wandb(cfg, logging_config, model_bundle.model, total_params, trainable_params)

    # Determine algorithm name from config or infer it
    # Assuming alg_config has a 'name' attribute or we can get it from cfg.algorithm.name if it exists
    alg_name = getattr(alg_config, "name", "ppo") # Default to ppo if not specified
    
    # Instantiate Algorithm
    AlgorithmClass = get_algorithm(alg_name)
    algorithm = AlgorithmClass(alg_config)
    
    trainer = Trainer(
        algorithm=algorithm,
        model=model_bundle.model,
        env_constructors=[env_fn],
        alg_config=alg_config,
        logging_config=logging_config,
        eval_config=eval_config,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        seed=cfg.seed,
        eval_location=eval_location,
    )
    trainer.train()


def run_training(cfg: DictConfig):
    hydra_output_dir = Path.cwd()
    try:
        hydra_output_dir = Path(HydraConfig.get().runtime.output_dir)
        os.environ["HYDRA_RUNTIME_OUTPUT_DIR"] = str(hydra_output_dir)
    except Exception:
        pass

    if cfg.wandb.enabled:
        wandb.init(
            project=cfg.wandb.project,
            config=OmegaConf.to_container(cfg, resolve=True),
            name=cfg.wandb.name,
            group=cfg.wandb.group,
            dir=cfg.wandb.dir,
            tags=cfg.wandb.tags,
        )

        with open_dict(cfg):
            for fname in ["git_sha.txt", "git_diff.patch", "git_dirty.txt"]:
                git_file = hydra_output_dir / fname
                if git_file.exists():
                    wandb.save(str(git_file))

    set_global_seeds(cfg.seed, cfg.deterministic_torch)
    configure_training(cfg)

    if cfg.wandb.enabled:
        wandb.finish()


@hydra.main(config_path="conf", config_name="8x8x128_dynamic.yaml", version_base=None)
def main(cfg: DictConfig):
    run_training(cfg)


if __name__ == "__main__":
    main()
