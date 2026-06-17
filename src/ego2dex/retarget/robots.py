"""Robot hand configurations for retargeting.

A :class:`RobotConfig` says *which robot links are the task/fingertip targets*
and *which joints are optimized*, plus the human-keypoint indices (standard-21)
each target link should match. DOF is NEVER assumed equal to the human hand --
the optimizer's link/joint mapping absorbs the mismatch (human 21 kpts / MANO 45
-> robot 16 or 21).

The **ORCA hand (16 active DOF)** is a first-class entry pointing at a
user-provided URDF (from the user's ``orca_sim`` / orcahand assets). Allegro,
Shadow and LEAP reuse ``dex_retargeting``'s bundled configs/URDFs.

Standard-21 indices (see ego2dex.topology): wrist=0; tips = 4,8,12,16,20;
MCPs = 5,9,13,17 (+ thumb CMC=1).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from ..topology import FINGERTIP_INDICES

# Human (standard-21) indices for the 5 fingertips and the wrist.
TIP_HUMAN_INDICES = list(FINGERTIP_INDICES)  # [4, 8, 12, 16, 20]
WRIST_HUMAN_INDEX = 0


@dataclass
class RobotConfig:
    """One robot hand's retargeting setup."""

    name: str
    urdf_path: str | None = None
    retargeting_type: str = "position"  # position | vector | dexpilot
    num_dof: int | None = None
    # dex_retargeting built-in identifier (RobotName value), if applicable.
    builtin: str | None = None
    hand_type: str = "right"  # right | left
    # Robot link names that should reach the human fingertips (order matches
    # TIP_HUMAN_INDICES: thumb, index, middle, ring, pinky).
    finger_tip_link_names: list[str] = field(default_factory=list)
    wrist_link_name: str | None = None
    # Position-optimizer targets: robot links + the human indices they track.
    target_link_names: list[str] = field(default_factory=list)
    target_link_human_indices: list[int] = field(default_factory=list)
    # Joints to optimize (None -> all actuated joints in the URDF).
    target_joint_names: list[str] | None = None
    scaling_factor: float = 1.0
    low_pass_alpha: float = 0.2

    def resolved_urdf(self) -> str | None:
        """URDF path with ``${ENV}`` / ``~`` expanded; honors EGO2DEX_ORCA_URDF."""
        path = self.urdf_path
        if path is None and self.name == "orca":
            path = os.environ.get("EGO2DEX_ORCA_URDF")
        if path is None:
            return None
        return os.path.expanduser(os.path.expandvars(path))


# --------------------------------------------------------------------------- #
# Presets
# --------------------------------------------------------------------------- #
def orca_config(urdf_path: str | None = None, retargeting_type: str = "position") -> RobotConfig:
    """ORCA hand (16 active DOF). URDF is USER-SUPPLIED (orca_sim / orcahand).

    Link names below are placeholders matching the typical orcahand URDF naming
    -- verify them against your URDF (``<link name="...">``) and override via the
    retarget config's ``params`` if they differ.
    """
    tips = ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"]
    return RobotConfig(
        name="orca",
        urdf_path=urdf_path,
        retargeting_type=retargeting_type,
        num_dof=16,
        builtin=None,  # not bundled with dex_retargeting -> custom config dict
        finger_tip_link_names=tips,
        wrist_link_name="palm",
        target_link_names=tips,
        target_link_human_indices=TIP_HUMAN_INDICES,
        scaling_factor=1.0,
        low_pass_alpha=0.2,
    )


def _builtin(name: str, dof: int, robotname: str) -> RobotConfig:
    return RobotConfig(name=name, num_dof=dof, builtin=robotname)


ROBOTS: dict[str, RobotConfig] = {
    "orca": orca_config(),
    "allegro": _builtin("allegro", 16, "allegro"),
    "shadow": _builtin("shadow", 24, "shadow"),
    "leap": _builtin("leap", 16, "leap"),
}


def get_robot_config(name: str, **overrides) -> RobotConfig:
    """Fetch a preset by name and apply field overrides (e.g. urdf_path)."""
    if name not in ROBOTS:
        raise KeyError(f"Unknown robot '{name}'. Known: {sorted(ROBOTS)}")
    base = ROBOTS[name]
    if name == "orca":
        base = orca_config(
            urdf_path=overrides.pop("urdf_path", base.urdf_path),
            retargeting_type=overrides.pop("retargeting_type", base.retargeting_type),
        )
    cfg = RobotConfig(**{**base.__dict__})
    for k, v in overrides.items():
        if hasattr(cfg, k) and v is not None:
            setattr(cfg, k, v)
    return cfg
