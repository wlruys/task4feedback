from dataclasses import dataclass, field
from typing import Union, Optional
from .base import AlgorithmConfig

@dataclass
class UnifiedConfig(AlgorithmConfig):
    name: str = "ppo"
    type: str = "on_policy" # on_policy, off_policy
    implementation: str = "torchrl"
    
    # Common
    workers: int = 1
    threads_per_worker: int = 1
    collect_device: str = "cpu"
    update_device: str = "cpu"
    storing_device: str = "cpu"
    max_grad_norm: float = 0.5
    gamma: float = 0.99

    # Collection
    graphs_per_collection: int = 10
    states_per_collection: int = 1920
    rollout_steps: int = 0 # 0 means use states_per_collection / workers
    num_collections: int = 1000
    timeout: int = 86400
    collector_reset_at_each_iter: bool = True  # Whether to reset environments each iteration
    
    # PPO Specific
    minibatch_size: int = 250
    epochs_per_collection: int = 4
    clip_eps: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.001
    val_coef: float = 0.5
    lmbda: float = 0.95
    normalize_advantage: bool = False
    value_norm: str = "l2"
    compile_policy: bool = False
    compile_update: bool = False
    compile_advantage: bool = False
    log_norms: bool = False
    advantage_type: str = "gae"
    bagged_policy: str = "uniform"
    target_kl: Optional[float] = None  # Early stop epoch if KL divergence exceeds this (e.g., 0.01-0.03)
    vtrace_use_lambda: bool = False  # Whether to use lambda in VTrace (for special cases)
    
    # SAC/DQN Specific
    replay_buffer_size: int = 1000000
    batch_size: int = 256
    lr: float = 3e-4
    q_lr: float = 3e-4
    alpha_lr: float = 3e-4
    alpha_init: float = 1.0
    target_entropy: Union[str, float] = "auto"
    target_update_polyak: float = 0.995
    num_qvalue_nets: int = 2
    loss_function: str = "smooth_l1"
    updates_per_collection: int = 1
    
    # DQN Specific
    eps_init: float = 1.0
    eps_end: float = 0.01
    eps_decay: int = 100000
