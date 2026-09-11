# Environment Layer

## Architecture (target)

- `factory.make_env()` → creates `TrainEnv` (the only thing Trainer imports)
- `TrainEnv(gym.Wrapper)` → facade; delegates env-specific queries to `EnvBackend`
- `TrainEnv` → owns optional `VideoRecorder` (composition, not inheritance)
- `EnvBackend` → Protocol: `agent_pos`, `grid_width`, `grid_height`, `get_visible_cells()`
- `MinigridBackend` implements `EnvBackend` (wraps `MiniGridEnv`)
- `MemoryGymBackend` implements `EnvBackend` (future)
- `VideoRecorder` → owns `OverlayRender` + `AgentViewRenderer`
- `VideoRecorder` → handles frame buffers, start/stop, episode triggers, h264 encoding

## File layout

```
env/
├── factory.py                  # make_env() → TrainEnv
├── facade.py                   # TrainEnv + EnvBackend Protocol
├── minigrid_backend.py         # MinigridBackend
├── video/
│   ├── recorder.py             # VideoRecorder
│   ├── overlay_render.py       # OverlayRender (RF overlay compositing)
│   └── agent_view_renderer.py  # AgentViewRenderer (perceived-pos / timeline panel)
├── custom_env_wrappers.py      # LEGACY – being replaced by the above
└── pom_env.py                  # DEPRECATED
```
