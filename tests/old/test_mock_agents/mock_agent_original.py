

import torch
from tests.old.test_mock_agents.original_implementation_adapted import Agent as AgentOriginal

class MockAgentOriginal(AgentOriginal):
    
    # used to retrieve back the instance 
    instances = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)   

        self.instances.append(self)
        
        self.args_sequence = []
        self.returns_sequence = []

        self.action_and_value_calls = 0
        self.value_calls = 0


    def get_value(self, x, memory, memory_mask, memory_indices):

        self.value_calls += 1

        self.args_sequence.append({
            "function_name": "get_value",
            "x": x.cpu().clone(),
            "memory_frames": memory.cpu().clone(),
            "memory_mask": memory_mask.cpu().clone(),
            "memory_indices": memory_indices.cpu().clone(),
            "in_rng_state": torch.get_rng_state()
        })

        value = super().get_value(x, memory, memory_mask, memory_indices)
        res_dict = {"value": value.cpu().clone(), "out_rng_state": torch.get_rng_state()}

        self.returns_sequence.append(res_dict)

        return value

    def get_action_and_value(self, x, memory, memory_mask, memory_indices, action=None):
        self.action_and_value_calls += 1
        x_in = x.clone()

        self.args_sequence.append({
            "function_name": "get_action_and_value",
            "x": x_in.cpu().clone(),
            "memory_frames": memory.cpu().clone(),
            "memory_mask": memory_mask.cpu().clone(),
            "memory_indices": memory_indices.cpu().clone(),
            "action": action.cpu().clone() if action is not None else None,
            "in_rng_state": torch.get_rng_state()
        })

        action, log_probs, entropies, critic, memory, attn = \
            super().get_action_and_value(x, memory, memory_mask, memory_indices, action)
        

        #x_out, new_memory_frame, attention_weights = self.transformer(x_in,memory, 
        #                                                    memory_mask=memory_mask, 
        #                                                    memory_indices=memory_indices)
        #x_out = self.hidden_post_trxl(x_out)




        result_dict = {
            'action': action.cpu().clone(),
            'log_probs': log_probs.cpu().clone(),
            'entropies': entropies.cpu().clone(),
            'critic': critic.cpu().clone(),
            'memory': memory.cpu().clone(),
            'attn': [a.cpu().clone() for a in attn],
            #'x_out': x_out.cpu(),
            'out_rng_state': torch.get_rng_state()
        }
        
        self.returns_sequence.append(result_dict)

        return action, log_probs, entropies, critic, memory, attn
