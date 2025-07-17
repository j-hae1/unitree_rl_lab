from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def bad_pelvis_ori(
    env: ManagerBasedRLEnv,
    limit_euler_angle: List[float] = [0.8, 1.0],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the asset's orientation is out of predefined range

    Args:
        limit_euler_angle: euler angle threshold [roll, pitch]. Episode
            will be terminated if the abs of the root euler angle will
            exceed this threshold
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # torso_idx = asset.find_bodies(torso_body_name)[0][0]
    euler = math_utils.wrap_to_pi(
        torch.stack(math_utils.euler_xyz_from_quat(asset.data.root_quat_w), dim=-1)
    )
    out_of_limit = torch.logical_or(
        torch.abs(euler[..., 0]) > limit_euler_angle[0],
        torch.abs(euler[..., 1]) > limit_euler_angle[1],
    )

    return out_of_limit