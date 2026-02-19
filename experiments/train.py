import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import wandb
from hydra.utils import instantiate
from task4feedback.experiment_helper.graph import make_graph_builder
from task4feedback.experiment_helper.env import make_env
from task4feedback.experiment_helper.model import (
    create_td_actor_critic_models,
    load_policy_from_checkpoint,
)
from task4feedback.experiment_helper.algorithm import (
    create_optimizer,
    create_lr_scheduler,
)
from task4feedback.experiment_helper.run_name import make_folder_name

from task4feedback.ml.algorithms.ppo import run_ppo
from task4feedback.interface.wrappers import *
from task4feedback.ml.models import *

# torch.multiprocessing.set_sharing_strategy("file_descriptor")
# torch.multiprocessing.set_sharing_strategy("file_system")
from pathlib import Path
import os
from hydra.core.hydra_config import HydraConfig
import torch
import numpy
import random
import pickle


def configure_training(cfg: DictConfig, normalization=None):
    # start_logger()
    run_name, _, _, _ = make_folder_name(cfg)
    graph_builder = make_graph_builder(cfg)
    if normalization is None:
        env, normalization = make_env(graph_builder=graph_builder, cfg=cfg)
        norm_dir = os.path.join("./norms", run_name)
        os.makedirs(norm_dir, exist_ok=True)

        with open(
            os.path.join(norm_dir, f"{cfg.feature.observer.version}_norm.pkl"), "wb"
        ) as f:
            pickle.dump(normalization, f)
    else:
        env = make_env(
            graph_builder=graph_builder, cfg=cfg, normalization=normalization
        )
    observer = env.get_observer()
    feature_config = FeatureDimConfig.from_observer(observer)
    model, reference, lstm = create_td_actor_critic_models(cfg, feature_config)

    if cfg.get("load_policy", None):
        ckpt_path = Path(cfg.load_policy)
        print(f"Loading policy from checkpoint: {ckpt_path}")
        loaded = load_policy_from_checkpoint(model, ckpt_path)
        assert loaded, f"Failed to load model from {ckpt_path}"
        print(f"Successfully loaded model from {ckpt_path}")
        exit()
    # ckpt_path = Path("/home/cc/task4feedback_torchrl/experiments/saved_models_test")
    # folder_name, _, _, _ = make_folder_name(cfg)
    # model_path = Path("./") / folder_name
    # files = list(model_path.glob("*.pt"))
    # assert len(files) <= 1, f"Multiple checkpoint files found in {model_path}"
    # if len(files) == 1:
    #     ckpt_path = files[0]
    #     print(f"Loading policy from checkpoint: {ckpt_path}")
    #     loaded = load_policy_from_checkpoint(model, ckpt_path)
    #     assert loaded, f"Failed to load model from {ckpt_path}"
    # model.actor.load_state_dict(torch.load("/home/cc/task4feedback_torchrl/experiments/dataset/8w_256lvl_4gpu_noise_10-1_72GB/bc_actor_best_eft.pt", weights_only=False))

    def env_fn(eval: bool = False, imported_cfg: DictConfig = None):
        if imported_cfg is not None:
            return make_env(
                graph_builder=make_graph_builder(imported_cfg),
                cfg=imported_cfg,
                lstm=lstm,
                normalization=normalization,
                eval=eval,
            )
        else:
            return make_env(
                graph_builder=graph_builder,
                cfg=cfg,
                lstm=lstm,
                normalization=normalization,
                eval=eval,
            )

    alg_config = instantiate(cfg.algorithm)

    optimizer = create_optimizer(cfg)
    lr_scheduler = create_lr_scheduler(cfg)
    logging_config = instantiate(cfg.logging)

    eval_config = instantiate(cfg.eval)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if cfg.wandb.enabled:
        wandb.run.summary["model/parameters_total"] = int(total_params)
        wandb.run.summary["model/parameters_trainable"] = int(trainable_params)
        try:
            arch_file = Path(HydraConfig.get().runtime.output_dir) / "model_arch.txt"
            arch_file.write_text(str(model))
            wandb.save(str(arch_file))
        except Exception as e:
            print(f"Failed to save model architecture: {e}")
        try:
            wandb.watch(model, log="all")
        except Exception as e:
            print(f"wandb.watch failed: {e}")

    run_ppo(
        actor_critic_module=model,
        env_constructors=[env_fn],
        logging_config=logging_config,
        ppo_config=alg_config,
        eval_config=eval_config,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        seed=cfg.seed,
        resume_from=cfg.resume_from,
    )


@hydra.main(config_path="conf", config_name="static_batch.yaml", version_base=None)
def main(cfg: DictConfig):
    # cfg.graph.config.workload_args.traj_type exist
    if "Dilation" in cfg.network.layers.state._target_:
        if "Uncond" in cfg.network.layers.state._target_:
            network = "UncondCNN"
        else:
            network = "CNN"
    elif "Vector" in cfg.network.layers.state._target_:
        network = "Vector"
    elif "GNN" in cfg.network.layers.state._target_:
        network = "GNN"
    else:
        print(cfg.network.layers.state._target_)
        raise ValueError("Unknown network type in cfg.network.layers.state._target_")

    if cfg.graph.mesh._target_ == "task4feedback.graphs.mesh.generate_quad_mesh":
        run_name, _, _, _ = make_folder_name(cfg)

        best_policy_dir = Path(cfg.wandb.dir).parent / "best_policies" / f"{run_name}"
        checkpoint_dir = Path(cfg.wandb.dir).parent / "checkpoints" / f"{run_name}"

        cfg.eval.pickle_path = (
            f"./pickled_evaluation/{cfg.feature.observer.version}/{run_name}.pkl"
        )
        cfg.eval.expert_path = (
            f"./dataset/{run_name}/{cfg.eval.expert_path}.pkl"
            if cfg.eval.expert_path is not None
            else None
        )
        norm_path = f"./norms/{run_name}/{cfg.feature.observer.version}_norm.pkl"

        if not os.path.exists(cfg.eval.pickle_path):
            print(f"Pickle path {cfg.eval.pickle_path} does not exist.")
            cfg.eval.pickle_path = None
        else:
            print(f"Using pickle path {cfg.eval.pickle_path}")

        if not os.path.exists(cfg.eval.expert_path):
            print(f"Expert path {cfg.eval.expert_path} does not exist.")
            cfg.eval.expert_path = None
        else:
            print(f"Using expert path {cfg.eval.expert_path}")

        if os.path.exists(norm_path):
            print(f"Loading normalization from {norm_path}")
            normalization = pickle.load(open(norm_path, "rb"))
        else:
            normalization = None

        # Make a dir if not exists
        best_policy_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        cfg.logging.best_policy_dir = str(best_policy_dir)
        cfg.logging.checkpoint_dir = str(checkpoint_dir)

        print(f"Best Policy dir: {cfg.logging.best_policy_dir}")
        cfg.logging.best_policy_name = f"{cfg.feature.observer.version}"
        print(f"Best Policy name: {cfg.logging.best_policy_name}")

    if cfg.wandb.enabled:
        wandb.init(
            project=cfg.wandb.project,
            config=OmegaConf.to_container(cfg, resolve=True),
            name=cfg.wandb.name,
            group=cfg.wandb.group,
            dir=cfg.wandb.dir,
            tags=cfg.wandb.tags,
        )

        hydra_output_dir = Path(HydraConfig.get().runtime.output_dir)

        with open_dict(cfg):
            for fname in ["git_sha.txt", "git_diff.patch", "git_dirty.txt"]:
                git_file = hydra_output_dir / fname
                if git_file.exists():
                    wandb.save(str(git_file))

    torch.manual_seed(cfg.seed)
    numpy.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.use_deterministic_algorithms(cfg.deterministic_torch)

    configure_training(cfg, normalization=normalization)

    if cfg.wandb.enabled:
        wandb.finish()


if __name__ == "__main__":
    main()
