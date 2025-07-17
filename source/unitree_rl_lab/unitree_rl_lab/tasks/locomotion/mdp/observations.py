from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def gait_phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf"):
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    global_phase = (env.episode_length_buf * env.step_dt) % period / period

    # phase = torch.zeros(env.num_envs, 2, device=env.device)
    # phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    # phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    phase_left = global_phase  # shape: N
    phase_right = (global_phase + 0.5) % 1.0  # Assuming a 50% offset for the right leg, shape : N

    phase_both = torch.cat([phase_left.unsqueeze(1), phase_right.unsqueeze(1)], dim=1)  # shape: N, 2

    # sin and cos phase_both (N, 2) -> N, 4
    phase = torch.cat([torch.sin(phase_both * 2 * torch.pi), torch.cos(phase_both * 2 * torch.pi)], dim=-1)

    return phase
