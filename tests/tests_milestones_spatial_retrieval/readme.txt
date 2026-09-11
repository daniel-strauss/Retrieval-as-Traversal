In docs/plan_implement_spatial_retrieval, 7 milestones are mentioned for the implementation of spatial retrieval. In this folder we write the tests for each milestone before implementing them. 

----
This test suite has a recording of rewards and losses for the code before the implementation started for the two configs in tests/test_milestones_spatial_retrieval/configs. 

Note that because we add the gausian position heads, we need to create them, with a priviate seed, to be able to reproduce the original results. Once the code reaches a state where the head actually has be used we remove the provate seed and discard these comparison tests.

To generate the recordings run 

python main.py --train_config tests/test_milestones_spatial_retrieval/configs/{name_of_config}

I will manually write down the reward trace of the first few steps into a tuple.

-----

Here is a milestone- test overview: 

M1: 
Test File
check that the reward trace for the first 10 epochs is equal, no test needed
M2: 
Test File: 
- After a rollout, check memory_rollout_buffer.data.perceived_positions has valid values.
- Positions should be zero when env_step=0 (agent is at spawn).
- Positions should change as the agent moves.
- No change to training curves or loss.
###################
Out of test file:
- agent view is added to environment renderer
it gets the agents preceived coordinates and renders next to the regular video 
another video that shows the trajectory
- agent view also renders the receptive fields relative to its own position
- it will render in the future positional grid cell encodings
- it may also render reconstructions
##################### 
M3:
Test File:
Unit test: write 5 positions across 2 envs, query back, verify correct env_steps returned.
Unit test: reset env 0, verify its index is empty, env 1 still has data.
Unit test: write same position multiple times, read with k=2, verify most recent 2 returned.
Integration: run a short rollout, inspect spatial index contents, verify they match agent positions.
No change to training.