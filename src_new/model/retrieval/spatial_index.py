import torch


class SpatialIndex:
    """
    Class for spatial indexing of states in the memory segment.

    It's purpose:
     - you provide a (position in space, agent) pairs, it gives you (agent, all env steps), when
        the agent was at that position in space


    For this to work you need to
    - call add_fov with a spawn-relative fov at every step
    - reset envs

    """

    store_as_tensor = True  # whether to store the spatial index as a tensor for faster retrieval,
    # or as a dict for higher memory efficiency.

    def __init__(
        self,
        num_envs: int,
        grid_w: int,
        grid_h: int,
        device: torch.device,
        max_episode_steps: int = 0,
    ):
        self.N = num_envs
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.device = device
        self.W = grid_w * 2 - 1
        self.H = grid_h * 2 - 1

        # the step of the next write for each env, used for sanity
        self.next_write_step: torch.Tensor = torch.zeros(self.N, dtype=torch.long, device=device)

        if self.store_as_tensor:
            if max_episode_steps <= 0:
                raise ValueError("max_episode_steps must be > 0 when store_as_tensor=True")
            self.max_episode_steps = max_episode_steps
            # (N, W, H, T) bool — fov_history[n, i, j, t] = True means env n
            # saw spawn-relative cell (i - grid_w+1, j - grid_h+1) at step t.
            self.fov_history = torch.zeros(
                (num_envs, self.W, self.H, max_episode_steps),
                dtype=torch.bool,
                device=device,
            )
        else:
            # (agent, spawn-relative x, spawn-relative y) -> set of env steps
            self.spatial_index: dict[int, dict[tuple[int, int], set[int]]] = {
                agent: {
                    (x, y): set()
                    for x in range(-(grid_w - 1), grid_w)
                    for y in range(-(grid_h - 1), grid_h)
                }
                for agent in range(self.N)
            }

    def add_fov(self, perceived_fov: torch.Tensor, env_steps: torch.Tensor):
        """Record visible cells in spawn-relative coordinates.

        Args:
            perceived_fov: (num_envs, 2*grid_w-1, 2*grid_h-1) bool tensor.
                Index (i, j) represents spawn-relative position
                (i - (grid_w-1), j - (grid_h-1)).
            env_steps: (num_envs,) current environment steps.
        """
        if not torch.all(env_steps == self.next_write_step):
            raise ValueError(
                "Env steps must be sequential and match the expected next steps for "
                f"each environment. {env_steps=} vs {self.next_write_step=}"
            )

        if self.store_as_tensor:
            # scatter perceived_fov into the time-step slice for each env
            idx = env_steps.view(self.N, 1, 1, 1).expand(self.N, self.W, self.H, 1)
            self.fov_history.scatter_(3, idx, perceived_fov.unsqueeze(3))
        else:
            ox, oy = self.grid_w - 1, self.grid_h - 1
            for agent, ti, tj in zip(*torch.where(perceived_fov)):
                x = int(ti.item()) - ox
                y = int(tj.item()) - oy
                self.spatial_index[int(agent.item())][(x, y)].add(int(env_steps[agent]))

        self.next_write_step += 1

    def reset_envs(self, dones: torch.Tensor):
        self.next_write_step[dones] = 0
        if self.store_as_tensor:
            self.fov_history[dones] = False
        else:
            for agent in torch.where(dones)[0]:
                self.spatial_index[int(agent)] = {
                    (x, y): set()
                    for x in range(-(self.grid_w - 1), self.grid_w)
                    for y in range(-(self.grid_h - 1), self.grid_h)
                }

    def get_memory_indices_and_mask(
        self, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        By k we denote the highest number of memories for a given retrieved position.

        Args:
            positions (torch.Tensor): A tensor of shape (num_envs, 2) representing
                the positions in the environment for which to retrieve memory indices and masks.

        Returns:
            - memory_indices (torch.Tensor): A tensor of shape (num_envs, k)
                  containing the memory indices for each environment.
            - memory_mask (torch.Tensor): A tensor of shape (num_envs, k) containing the memorys
                which memory_indices are valid
        """
        if positions.dtype != torch.long:
            raise ValueError(f"Positions tensor must be of dtype torch.long, got {positions.dtype}")

        if self.store_as_tensor:
            return self._get_indices_tensor(positions)
        return self._get_indices_dict(positions)

    # ── tensor path ──────────────────────────────────────────────────────

    def _get_indices_tensor(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ox, oy = self.grid_w - 1, self.grid_h - 1
        xi = positions[:, 0] + ox
        yi = positions[:, 1] + oy
        if (xi < 0).any() or (xi >= self.W).any() or (yi < 0).any() or (yi >= self.H).any():
            raise ValueError(
                f"Positions out of bounds: x must be in [{-ox}, {ox}], "
                f"y must be in [{-oy}, {oy}], got {positions}"
            )

        env_arange = torch.arange(self.N, device=self.device)
        # hits[n, t] = True  ⇔  env n saw the queried cell at step t
        hits = self.fov_history[env_arange, xi, yi]  # (N, max_episode_steps)

        k = int(hits.sum(dim=1).max().item())
        if k == 0:
            return (
                torch.zeros(self.N, 0, dtype=torch.long, device=self.device),
                torch.zeros(self.N, 0, dtype=torch.bool, device=self.device),
            )

        # cumsum gives each True entry its 0-based position in the output row
        pos_in_output = hits.cumsum(dim=1) - 1  # (N, max_episode_steps)
        step_indices = (
            torch.arange(self.max_episode_steps, device=self.device).unsqueeze(0).expand(self.N, -1)
        )

        memory_indices = torch.full((self.N, k), -1, dtype=torch.long, device=self.device)
        memory_mask = torch.zeros(self.N, k, dtype=torch.bool, device=self.device)

        env_exp = env_arange.unsqueeze(1).expand_as(pos_in_output)
        memory_indices[env_exp[hits], pos_in_output[hits]] = step_indices[hits]
        memory_mask[env_exp[hits], pos_in_output[hits]] = True

        return memory_indices, memory_mask

    # ── dict path (original) ─────────────────────────────────────────────

    def _get_indices_dict(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        per_env_steps: list[list[int]] = []
        for agent_idx in range(self.N):
            x = int(positions[agent_idx, 0].item())
            y = int(positions[agent_idx, 1].item())
            steps = sorted(self.spatial_index[agent_idx].get((x, y), set()))
            per_env_steps.append(steps)

        k = max(len(s) for s in per_env_steps) if per_env_steps else 0

        if k == 0:
            memory_indices = torch.zeros(self.N, 0, dtype=torch.long, device=self.device)
            memory_mask = torch.zeros(self.N, 0, dtype=torch.bool, device=self.device)
            return memory_indices, memory_mask

        memory_indices = torch.full((self.N, k), -1, dtype=torch.long, device=self.device)
        memory_mask = torch.zeros(self.N, k, dtype=torch.bool, device=self.device)

        for i, steps in enumerate(per_env_steps):
            n = len(steps)
            if n > 0:
                memory_indices[i, :n] = torch.tensor(steps, dtype=torch.long)
                memory_mask[i, :n] = True

        return memory_indices, memory_mask

    # ── inverse retrieval (needed for plotting) ─────────────────────────────────────────────

    def get_cumulative_fov(self, env_id: int, up_to_t: int) -> torch.Tensor:
        """Return (W, H) bool mask of all cells ever visible up to step *up_to_t* (inclusive)."""
        if not self.store_as_tensor:
            raise NotImplementedError("get_cumulative_fov requires store_as_tensor=True")
        if up_to_t < 0:
            return torch.zeros(self.W, self.H, dtype=torch.bool, device=self.device)
        return self.fov_history[env_id, :, :, : up_to_t + 1].any(dim=-1)

    def get_retrieved_fov_union(
        self, env_id: int, indices: torch.Tensor, masks: torch.Tensor
    ) -> torch.Tensor:
        """Return (W, H) bool mask of cells visible during the given valid steps.

        Args:
            env_id: Which environment to query.
            indices: (k,) env-step indices.
            masks: (k,) bool mask indicating which indices are valid.

        Returns:
            (W, H) bool tensor — True where the agent's FOV covered that cell
            at any of the valid steps.
        """
        if self.store_as_tensor:
            valid_steps = indices[masks.bool()]
            if len(valid_steps) == 0:
                return torch.zeros(self.W, self.H, dtype=torch.bool, device=self.device)
            return self.fov_history[env_id, :, :, valid_steps].any(dim=-1)

        else:
            fov_union = torch.zeros(self.W, self.H, dtype=torch.bool, device=self.device)
            for step, mask in zip(indices, masks):
                if mask:
                    fov_union |= self.fov_history[env_id][step]
            return fov_union
