import numpy as np
import torch
from minigrid.minigrid_env import MiniGridEnv

                        # next memory, 1, memory index



def get_visible_cells(env: MiniGridEnv) -> torch.Tensor:
    """Return a boolean mask of shape (width, height) indicating which grid
    cells are currently visible to the agent.

    MiniGrid-specific: uses ``gen_obs_grid`` for per-cell visibility and
    ``get_view_exts`` to map view-local coordinates to absolute grid
    coordinates.

    Args:
        env: A gymnasium environment whose ``unwrapped`` attribute is a
            ``MiniGridEnv``.

    Returns:
        Boolean torch tensor of shape ``(grid_width, grid_height)``.
    """
    
    grid_w, grid_h = env.width, env.height
    view_size = env.agent_view_size

    # Visibility mask in view-local coordinates
    _, vis_mask = env.gen_obs_grid(view_size)

    # Direction vectors for coordinate transform
    ax, ay = env.agent_pos
    dx, dy = env.dir_vec
    rx, ry = env.right_vec
    hs = view_size // 2

    # Top-left of view in absolute coords — must match get_view_coords
    tx = ax + dx * (view_size - 1) - rx * hs
    ty = ay + dy * (view_size - 1) - ry * hs

    # faster alternative
    locations = np.array([[rx, -dx], 
                          [ry, -dy]]) @ np.argwhere(vis_mask).T + np.array([[tx], [ty]]) 
    locations = locations.T
    mask = torch.zeros((grid_w, grid_h), dtype=torch.bool)
    mask[locations[:, 0], locations[:, 1]] = True
    
    return mask

