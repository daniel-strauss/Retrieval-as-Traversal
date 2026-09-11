
"""
This test goes like this:

MockAgentOriginal: acts like the original implementation, with the same random seed and same initial weights. 
        all it does different is to record the get action and value calls arguments it get and the memory 
        indices and masks it produces, for each rollout step and environment.
MockAgentNew: acts like the new implementation, with the same random seed and same initial weights.
        all it does different is to record the get action and value calls arguments it get and the
"""


import os
from pathlib import Path
from unittest.mock import patch
import warnings

import torch

from main import _load_section
from tests.old.test_mock_agents import original_implementation_adapted 
from tests.old.test_mock_agents.original_implementation_adapted import Agent as AgentOriginal
from tests.old.test_mock_agents.original_implementation_adapted import Args as ArgsOriginal

from src_new.model.agent_module import Action, AgentModule as AgentNew
from src_new.config import Config as ConfigNew
from src_new.config import TrainConfig, LogConfig


from tests.old.test_mock_agents.in_out_checker import InOutChecker
from tests.old.test_mock_agents.mock_agent_original import MockAgentOriginal
from tests.old.test_mock_agents.mock_agent_new import MockAgentNew

# if you want to use cuda with deterministic rounding (otherwise masked memories will cause divergent 
# behaviour)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True)

# whethere to create a traceback even when the missmatch is only worthy a warning 
# (e.g. a masked value missmatch)
raise_at_warnings = True

# parameters that overwrite the actual training parameters
use_custom = True # orig: False
test_steps = 10000 if use_custom else 10000 # Original: 2048000
num_envs = 2 if use_custom else 16 # Original: 16
hidden_dim = 4 if use_custom else 256  # Original: 256
memory_length = 32 if use_custom else 64 # Original: 64
num_rollout_steps = 128 if use_custom else 256 # Original: 256 (Note bug may appear at 64)
num_minibatches = num_rollout_steps // 16 if use_custom else 8   # Original 8, (Original num steps 512, num envs=16)
cuda = True if use_custom else True # Original: True, set to False to avoid potential GPU masked-dependent round issues in the test.
# in the original implementation they set the as the last values entropy and they do not average
init_ent_coef = 0.0001 if use_custom else 0.0001
final_ent_coef = 0.00001 if use_custom else 0.000001


"""
Recordings to add:
- get_value
- reconstruct_observation (that both are not being called)

"""

###################
# Args for both versions
###################



args_original = ArgsOriginal(
    env_id="MiniGrid-MemoryS9-v0",
    total_timesteps=test_steps, # in our goto experiment is 2048000
    num_envs=num_envs, # Original: 16
    num_steps=num_rollout_steps,# originally 256
    trxl_num_layers=2,
    trxl_num_heads=4,
    trxl_dim=hidden_dim, # 256,
    trxl_memory_length=memory_length, # 64,
    max_grad_norm=0.25,
    anneal_steps=4096000,
    clip_coef=0.2,
    cuda=cuda, # original True
    init_ent_coef=init_ent_coef,
    final_ent_coef=final_ent_coef,
    num_minibatches=num_minibatches, # 8,
)


log = _load_section(Path("./configs/log_config/default.yaml"), LogConfig)
train = _load_section(
    Path("./configs/train_config/cleanrl_minigrid_example.yaml"),
TrainConfig,
    exclude={"batch_size", "minibatch_size", "num_iterations"},
)

args_new = ConfigNew(
    log=log,
    train=train
)

args_new.train.total_timesteps = test_steps
args_new.train.trxl_dim = hidden_dim
args_new.train.trxl_memory_length = memory_length
args_new.train.num_minibatches = num_minibatches
args_new.train.num_rollout_steps = num_rollout_steps
args_new.train.num_envs = num_envs
args_new.train.cuda = cuda
args_new.train.init_ent_coef = init_ent_coef
args_new.train.final_ent_coef = final_ent_coef

# TODO DEBUG REMOVE
args_new.train.use_old_bootstrap = True


args_new.train.exp_name = "test_mock_agents__new" 
args_original.exp_name = "test_mock_agents__original"


#check train vars are the same
common = set(vars(args_original)) & set(vars(args_new.train)) - {'exp_name'}
assert all(getattr(args_original, k) == getattr(args_new.train, k) for k in common), \
    f"Mismatch: {[k for k in common if getattr(args_original, k) != getattr(args_new.train, k)]}"






##########################
# Main test logic
##########################



def record_original_agent():
    with patch(
        "test_mock_agents.original_implementation_adapted.Agent",
        MockAgentOriginal
    ):
        original_implementation_adapted.main(args_original)

    if len(MockAgentOriginal.instances) != 1:
        raise ValueError(f"Expected exactly one instance of MockAgentOriginal, but got "
                         f"{len(MockAgentOriginal.instances)}")
    
    agent = MockAgentOriginal.instances[0]
    return agent 



def test_main():
    print("Recording original agent...")
    agent_original = record_original_agent()
    print("Finished recording original agent.")
    in_out_checker = InOutChecker(agent_original, raise_at_warnings=raise_at_warnings)

    print("Running new agent and comparing inputs/outputs...")
    with patch(
        "src_new.model.agent.AgentModule",
        lambda *args, **kwargs: MockAgentNew(in_out_checker, *args, **kwargs)
    ):
        from src_new.trainer.trainer import Trainer as TrainerNew
        trainer = TrainerNew(args_new)
        agent_new = trainer.agent.module
        try:
            trainer.run()
        finally:
            in_out_checker.close()

    if agent_new.action_and_value_calls != agent_original.action_and_value_calls:
        raise ValueError(f"Expected to check {agent_original.action_and_value_calls} calls to "\
                            f"get_action_and_value, but only checked {agent_new.action_and_value_calls}")
    if agent_new.value_calls != agent_original.value_calls:
        raise ValueError(f"Expected to check {agent_original.value_calls} calls to "
                         f"get_value, but only checked {agent_new.value_calls}")



    print(f"Finished running new agent. All inputs and outputs matched, num calls in agent_new: "\
          f"{agent_new.action_and_value_calls}")
    print(f"Logs written to {in_out_checker.output_dir}")



if __name__ == "__main__":
    test_main()

