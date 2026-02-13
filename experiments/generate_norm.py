import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from hydra.utils import instantiate

from task4feedback.experiment_helper.graph import make_graph_builder
from task4feedback.experiment_helper.env import make_env
from task4feedback.experiment_helper.model import create_td_actor_critic_models, load_policy_from_checkpoint
from task4feedback.experiment_helper.algorithm import create_optimizer, create_lr_scheduler
from task4feedback.experiment_helper.run_name import make_folder_name

from task4feedback.ml.algorithms.ppo import run_ppo, run_ppo_lstm
from task4feedback.interface.wrappers import *
from task4feedback.ml.models import *

# torch.multiprocessing.set_sharing_strategy("file_descriptor")
# torch.multiprocessing.set_sharing_strategy("file_system")

from hydra.experimental.callbacks import Callback
from hydra.core.utils import JobReturn
from omegaconf import DictConfig, open_dict
from pathlib import Path
import git
import os
from hydra.core.hydra_config import HydraConfig
from task4feedback.experiment_helper.run_name import make_run_name, cfg_hash

import torch
import numpy
import random
import pickle


def configure_training(cfg: DictConfig, normalization=None):
    # start_logger()
    run_name, _, _, _ = make_folder_name(cfg)
    graph_builder = make_graph_builder(cfg)
    env, normalization = make_env(graph_builder=graph_builder, cfg=cfg)
    norm_dir = os.path.join("./norms", run_name)
    os.makedirs(norm_dir, exist_ok=True)

    with open(os.path.join(norm_dir, f"{cfg.feature.observer.version}_norm.pkl"), "wb") as f:
        pickle.dump(normalization, f)


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

        checkpoint_path = Path(cfg.wandb.dir).parent / "model_checkpoints" / f"{run_name}"
        cfg.eval.pickle_path = f"./pickled_evaluation/{cfg.feature.observer.version}/{run_name}.pkl"
        cfg.eval.expert_path = f"./dataset/{run_name}/{cfg.eval.expert_path}.pkl" if cfg.eval.expert_path is not None else None
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
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        cfg.logging.best_policy_dir = str(checkpoint_path)
        print(f"Best Policy dir: {cfg.logging.best_policy_dir}")
        cfg.logging.best_policy_name = f"{cfg.feature.observer.version}"
        print(f"Best Policy name: {cfg.logging.best_policy_name}")

    torch.manual_seed(cfg.seed)
    numpy.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.use_deterministic_algorithms(cfg.deterministic_torch)

    configure_training(cfg, normalization=normalization)


if __name__ == "__main__":
    main()
