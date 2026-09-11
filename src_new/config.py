from dataclasses import asdict, dataclass, field, fields
from typing import Optional


@dataclass
class LogConfig:
    """Logging and tracking settings."""

    track: bool = False
    """whether to track the experiment with Weights & Biases"""
    wandb_project_name: str = "cleanRL"
    """W&B project name"""
    wandb_entity: Optional[str] = None
    """W&B entity (team or user)"""
    capture_video: bool = False
    """whether to record videos of the agent's gameplay"""
    num_vids_per_trigger: int = 10
    """if episode n is triggered, record episodes n, n+1, ..., n+num_vids_per_trigger-1"""
    render_debug: bool = False
    """whether to render debug overlays"""
    show_receptive_field: bool = False
    """whether to visualise the transformer receptive field"""
    save_model: bool = False
    """whether to save the trained model to disk"""


@dataclass
class TrainConfig:
    """Training, environment and model hyperparameters."""

    # ── Experiment identity ───────────────────────────────────────────────────
    exp_name: str = "trainer"
    """name of this experiment run"""
    seed: int = 1
    """random seed for reproducibility"""
    torch_deterministic: bool = True
    """whether to use deterministic algorithms in PyTorch"""
    cuda: bool = True
    """whether to use CUDA when available"""

    # ── Environment ───────────────────────────────────────────────────────────
    env_id: str = "MortarMayhem-Grid-v0"
    """Gymnasium environment id"""
    total_timesteps: int = 200_000_000
    """total number of environment steps for the run"""

    # ── PPO hyperparameters ───────────────────────────────────────────────────
    init_lr: float = 2.75e-4
    """initial learning rate (annealed toward final_lr)"""
    final_lr: float = 1.0e-5
    """final (minimum) learning rate after annealing"""
    num_envs: int = 32
    """number of parallel environments"""
    num_rollout_steps: int = 512
    """number of rollout steps per environment per iteration"""
    anneal_steps: int = 32 * 512 * 10_000
    """number of global steps over which lr and ent_coef are annealed; 0 = no annealing"""
    gamma: float = 0.995
    """discount factor"""
    gae_lambda: float = 0.95
    """GAE lambda parameter"""
    num_minibatches: int = 8
    """number of minibatches per PPO epoch"""
    update_epochs: int = 3
    """number of PPO optimisation epochs per iteration"""
    norm_adv: bool = False
    """whether to normalise advantages per minibatch"""
    clip_coef: float = 0.1
    """PPO clipping coefficient (epsilon)"""
    clip_vloss: bool = True
    """whether to use clipped value loss"""
    init_ent_coef: float = 0.0001
    """initial entropy bonus coefficient"""
    final_ent_coef: float = 0.000001
    """final (minimum) entropy bonus coefficient after annealing"""
    vf_coef: float = 0.5
    """value function loss coefficient"""
    max_grad_norm: float = 0.25
    """maximum gradient norm for clipping"""
    target_kl: Optional[float] = None
    """target KL divergence for early stopping; None = disabled"""

    # ── Transformer-XL architecture ───────────────────────────────────────────
    trxl_num_layers: int = 3
    """number of Transformer-XL layers"""
    trxl_num_heads: int = 4
    """number of attention heads per layer"""
    trxl_dim: int = 384
    """model (embedding) dimension"""
    trxl_memory_length: int = 119
    """number of past tokens kept in the memory segment"""
    trxl_positional_encoding: str = "absolute"
    """positional encoding type: 'absolute', 'learned', or 'none'"""
    prior_pe: bool = False
    """
    if True, apply positional encoding at memory creation, if false after memory retrieval
    if you want to add egoocentric PE (not yet implemented), or use the original version, this should be set to False. 
    """
    reconstruction_coef: float = 0.0
    """coefficient for the observation reconstruction loss; 0 = disabled"""

    # --- Newly added settings for memory handling and value bootstrap ---
    use_old_bootstrap: bool = True
    """whether to use the old bootstrap method for value targets, which uses the value of the last state in the trajectory"""
    apply_memory_masks: bool = False
    """whether to apply memory masks to the memory segment before feeding it into the transformer."""
    exclude_out_of_episode_memories: bool = True
    """
    When memories are being loaded memories from episodes after the loaded episode do not leak in,
    but are being set to 0 instead. 
    """
    uniform_attention_on_fully_masked_rows: bool = True
    """whether to use uniform attention when all entries in a row of the memory segment are masked (for exploration)"""
    hindsight_at_0: bool = False
    """whether to set all memory masks to True, where the env, global step is the first env step retrieving all future memories for that step """
    grid_cell_encoding: bool = False
    """whether to use grid cell encoding for the spatial memory module"""
    grid_cell_encoding_weight: float = 1.0
    """weight applied to the grid cell positional encoding relative to the temporal PE"""

    # --- Loading  Pretraied Models ---
    checkpoint_path: Optional[str] = None
    """path to a checkpoint to load submodules from; None = train from scratch"""
    checkpoint_submodules: list[str] = field(default_factory=list)
    """
    which submodules to load from the checkpoint available: 
    ["en
        if self.positional_encoding == "absolute":
coder", "transformer", "hidden_post_trxl", "actor_branches", "internal_head", "critic",  
    "transposed_cnn"]
    """

    # -- Spatial memory settings (if enabled) --
    use_spatial_memory: bool = False
    """whether to use the spatial memory module."""
    spatial_head_type: str = "gaussian"
    """which internal action head to use: 'gaussian', 'categorical_xy', 'position_moving'"""
    internal_ent_coef: float = 0.01
    """entropy bonus coefficient for the internal (retrieval) action"""
    retrieval_reward_coef: float = 0.0
    """coefficient for the retrieval-hit REINFORCE loss on the internal head; 0 = disabled"""
    internal_head_hidden_size: int = 0
    """hidden layer size for internal action heads; 0 = no hidden layer"""
    internal_head_warmup_steps: int = -1
    """number of global steps during which internal-head gradients are detached from the backbone;
    -1 = disabled (gradients always flow)"""


@dataclass
class Config:
    """Top-level config composed of a log sub-config and a train sub-config."""

    log: LogConfig
    train: TrainConfig

    def to_wandb_config(self) -> dict:
        """Return a flat dict suitable for wandb.init(config=...)."""
        return {**asdict(self.train), **asdict(self.log)}

    def apply_wandb_overrides(self, wandb_config) -> None:
        """Pull sweep overrides from wandb.config back into the dataclass fields."""
        for f in fields(self.train):
            if f.name in wandb_config:
                setattr(self.train, f.name, wandb_config[f.name])
        for f in fields(self.log):
            if f.name in wandb_config:
                setattr(self.log, f.name, wandb_config[f.name])
