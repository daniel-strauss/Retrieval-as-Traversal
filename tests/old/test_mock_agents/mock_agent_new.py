import torch

from src_new.memory.types import MemoryWindow
from src_new.model.agent_module import Action
from src_new.model.agent_module import AgentModule as AgentNew
from src_new.trainer.forward_diagnostics import ForwardDiagnostics
from tests.old.test_mock_agents.in_out_checker import InOutChecker


class MockAgentNew(AgentNew):
    # used to retrieve back the instance
    instances = []

    def __init__(self, in_out_checker: InOutChecker, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.instances.append(self)

        self.in_out_checker = in_out_checker
        self.check_func = in_out_checker.check_are_different_CLEAN  # OLD

        self.action_and_value_calls = 0
        self.value_calls = 0

    def get_value(
        self,
        x: torch.Tensor,
        memory_window: MemoryWindow,
        envs_t: torch.Tensor,
        perceived_pos: torch.Tensor,
    ) -> torch.Tensor:

        memory_frames = memory_window.frames
        memory_mask = memory_window.masks
        memory_indices = memory_window.indices_env  # lets hope that shit still the same

        self.value_calls += 1
        args_dict = {
            "function_name": "get_value",
            "x": x.cpu(),
            "memory_frames": memory_frames.cpu(),
            "memory_mask": memory_mask.cpu(),
            "memory_indices": memory_indices.cpu(),
            "in_rng_state": torch.get_rng_state(),
        }

        value = super().get_value(
            x, memory_window=memory_window, envs_t=envs_t, perceived_pos=perceived_pos
        )

        res_dict = {"value": value.cpu(), "out_rng_state": torch.get_rng_state()}

        self.check_func(args_dict, res_dict)

        return value

    def get_action_and_value(
        self,
        x: torch.Tensor,
        memory_window: MemoryWindow,
        envs_t: torch.Tensor,
        perceived_pos: torch.Tensor,
        action: Action | None = None,
        global_step: int | None = None,
        forward_diagnostics: ForwardDiagnostics | None = None,
    ) -> tuple[Action, torch.Tensor, torch.Tensor, list]:
        
        memory_frames=memory_window.frames
        memory_mask=memory_window.masks
        memory_indices=memory_window.indices_env

        self.action_and_value_calls += 1
        args_dict = {
            "function_name": "get_action_and_value",
            "x": x.cpu(),
            "memory_frames": memory_frames.cpu(),
            "memory_mask": memory_mask.cpu(),
            "memory_indices": memory_indices.cpu(),
            "action": action.external_action.cpu() if action is not None else None,
            "in_rng_state": torch.get_rng_state(),
        }

        action_pack, critic, memory, attn = super().get_action_and_value(
            x=x,
            memory_window=memory_window,
            envs_t=envs_t,
            perceived_pos=perceived_pos,
            action=action,
            global_step=global_step,
            forward_diagnostics=forward_diagnostics
        )

        result_dict = {
            "action": action_pack.external_action.cpu(),
            "log_probs": action_pack.external_log_probs.cpu(),
            "entropies": action_pack.external_entropy.cpu(),
            "critic": critic.cpu(),
            "memory": memory.cpu(),
            "attn": [a.cpu() for a in attn],
            "out_rng_state": torch.get_rng_state(),
        }

        self.check_func(args_dict, result_dict)

        return action_pack, critic, memory, attn


"""
UGLY BACKUP SHIT:


    def get_value__legacy(self, x, memory_frames, memory_mask, memory_indices):
        self.value_calls += 1
        args_dict = {
            "function_name": "get_value",
            "x": x.cpu(),
            "memory_frames": memory_frames.cpu(),
            "memory_mask": memory_mask.cpu(),
            "memory_indices": memory_indices.cpu(),
            "in_rng_state": torch.get_rng_state(),
        }

        value = super().get_value(x, memory_frames, memory_mask, memory_indices)

        res_dict = {"value": value.cpu(), "out_rng_state": torch.get_rng_state()}

        self.check_func(args_dict, res_dict)

        return value

        
      def get_action_and_value__legacy(
        self,
        x: torch.Tensor,
        memory_frames: torch.Tensor,
        memory_mask: torch.Tensor,
        memory_indices: torch.Tensor,
        action: Action | None = None,
    ):

        self.action_and_value_calls += 1
        args_dict = {
            "function_name": "get_action_and_value",
            "x": x.cpu(),
            "memory_frames": memory_frames.cpu(),
            "memory_mask": memory_mask.cpu(),
            "memory_indices": memory_indices.cpu(),
            "action": action.external_action.cpu() if action is not None else None,
            "in_rng_state": torch.get_rng_state(),
        }

        action_pack, critic, memory, attn = super().get_action_and_value(
            x, memory_frames, memory_mask, memory_indices, action
        )

        result_dict = {
            "action": action_pack.external_action.cpu(),
            "log_probs": action_pack.external_log_probs.cpu(),
            "entropies": action_pack.external_entropy.cpu(),
            "critic": critic.cpu(),
            "memory": memory.cpu(),
            "attn": [a.cpu() for a in attn],
            "out_rng_state": torch.get_rng_state(),
        }

        self.check_func(args_dict, result_dict)

        return action_pack, critic, memory, attn



"""
