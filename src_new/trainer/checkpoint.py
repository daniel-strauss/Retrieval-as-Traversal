from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import yaml

from src_new.config import TrainConfig
from src_new.model.agent_module import AgentModule
from src_new.trainer.metrics import Metrics


@dataclass
class CheckpointData:
    submodule_states: dict[str, dict]
    train_config: dict  # serialised TrainConfig
    metrics: dict[str, float]
    metadata: dict


class CheckpointManager:
    """
    Handles saving and loading of training checkpoints,
    including model weights, training config, and metrics.
    """

    base_path = "trained_models/trashy_models"

    SAVE_STRATEGIES = ["best"]

    def __init__(
        self, run_name: str, train_config: TrainConfig, save_strategy="best", cooldown: int = 15
    ):

        self.path = Path(f"{self.base_path}/{run_name}")

        self.train_config = train_config
        self.run_name = run_name

        # startegy for when to save
        if save_strategy not in self.SAVE_STRATEGIES:
            raise NotImplementedError(f"Save strategy: {save_strategy} not supported yet.")
        self.save_strategy = save_strategy
        self.best_metric = float("-inf")
        self.cooldown = cooldown
        self._iters_since_save = cooldown  # allow saving on first call

        self._initial_write()

    def _initial_write(self):
        """
        Writes creates the run directory and writes the train config.
        """

        self.path.mkdir(parents=False, exist_ok=False)
        self.path.joinpath("train_config.yaml").write_text(yaml.dump(asdict(self.train_config)))
        self.checkpoint_path = self.path.joinpath("checkpoints")
        self.checkpoint_path.mkdir(exist_ok=False)

    def checkpoint(
        self,
        module: AgentModule,
        metrics: Metrics,
        iteration: int,
    ):
        """
        Args:
            module: AgentModule containing the model to save
            metrics: Metrics object containing training metrics to save
        """

        self._iters_since_save += 1
        go_save_that_shit = False

        ###
        # Checkpointing strategy
        ###

        if self.save_strategy == "best":
            if metrics.episode_return_ma > self.best_metric + 1e-5:
                self.best_metric = metrics.episode_return_ma
                go_save_that_shit = True
        else:
            raise ValueError("Unsupported save strategy")

        ###
        # Cooldown
        ###

        if self._iters_since_save < self.cooldown:
            go_save_that_shit = False

        ###
        # Saving
        ###

        if go_save_that_shit:
            self._iters_since_save = 0
            checkpoint_name = f"i_{iteration}__r_{metrics.episode_return_ma:.3f}"
            local_path = self.checkpoint_path.joinpath(checkpoint_name)
            local_path.mkdir(exist_ok=False)

            local_path.joinpath("metrics.yaml").write_text(yaml.dump(metrics.as_dict()))
            torch.save(module.submodule_states(), local_path.joinpath("model_params"))

    @staticmethod
    def load(path: str, device: torch.device) -> dict[str, dict]:
        submodule_states = torch.load(path, map_location=device, weights_only=False)
        return submodule_states
