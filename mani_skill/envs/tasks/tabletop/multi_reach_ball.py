"""
MultiReachBall-v1 — Trivial sanity-check for HARL + ManiSkill3 pipeline.

Same scene as MultiPickBall (N Panda arms, each with a ball), but the task
is ONLY to move each TCP close to its ball.  No grasping, no placing.

If MAPPO/HAPPO can't solve this, the pipeline has a bug.
If they can, the pipeline works and MultiPickBall's difficulty is the issue.

Observation layout: identical to MultiPickBall (N*36 dims).
Reward: mean per-agent reach reward = 1 - tanh(5 * tcp_to_ball_dist)
        + 5.0 bonus if all TCPs within 0.05m of their balls
Success: all TCPs within 0.05m of their respective balls.
Episode length: 100 steps (shorter than MultiPickBall's 200).
"""

from typing import Any, List

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

from mani_skill.agents.robots.panda import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig

_ARM_QUAT = euler2quat(0, 0, np.pi / 2)
_ARM_Y_OFFSET = -0.615


@register_env("MultiReachBall-v1", max_episode_steps=100)
class MultiReachBall(BaseEnv):
    """N Panda arms each reach toward their own ball. No grasping required."""

    SUPPORTED_ROBOTS = [
        ("panda", "panda"),
        ("panda", "panda", "panda"),
        ("panda", "panda", "panda", "panda"),
    ]

    n_arms: int = 2
    arm_spacing: float = 0.4
    ball_radius: float = 0.02
    reach_thresh: float = 0.05
    robot_init_qpos_noise: float = 0.02

    def __init__(self, *args, n_arms: int = 2, robot_uids=None,
                 robot_init_qpos_noise: float = 0.02, **kwargs):
        self.n_arms = n_arms
        self.robot_init_qpos_noise = robot_init_qpos_noise
        if robot_uids is None:
            robot_uids = tuple(["panda"] * n_arms)
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25,
                max_rigid_patch_count=2**19,
                max_rigid_contact_count=2**21,
            )
        )

    @property
    def _default_sensor_configs(self):
        return []

    @property
    def _default_human_render_camera_configs(self):
        from mani_skill.sensors.camera import CameraConfig
        pose = sapien_utils.look_at([2.0, 0, 1.5], [0.0, 0.0, 0.3])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        poses = [
            sapien.Pose(p=[i * self.arm_spacing, _ARM_Y_OFFSET, 0], q=_ARM_QUAT)
            for i in range(self.n_arms)
        ]
        super()._load_agent(options, poses)

    def _load_scene(self, options: dict):
        floor_width = 500 if self.scene.parallel_in_single_scene else 20
        build_ground(self.scene, floor_width=floor_width, altitude=0)

        self.balls: List[Any] = []
        for i in range(self.n_arms):
            ball = actors.build_sphere(
                self.scene,
                radius=self.ball_radius,
                color=[1.0, 0.3 + 0.1 * i, 0.0, 1.0],
                name=f"ball_{i}",
                # Ball on the ground in front of arm
                initial_pose=sapien.Pose(p=[i * self.arm_spacing, 0, self.ball_radius]),
                body_type="kinematic",   # Ball is STATIC — no physics needed
                add_collision=False,
            )
            self.balls.append(ball)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)

            rest_qpos = np.array(
                [0.0, np.pi / 8, 0, -np.pi * 5 / 8, 0, np.pi * 3 / 4, np.pi / 4, 0.04, 0.04]
            )

            for i, sub_agent in enumerate(self.agent.agents):
                qpos = (
                    self._episode_rng.normal(0, self.robot_init_qpos_noise, (b, len(rest_qpos)))
                    + rest_qpos
                )
                qpos[:, 7:] = 0.04
                sub_agent.reset(qpos)
                sub_agent.robot.set_pose(
                    sapien.Pose(p=[i * self.arm_spacing, _ARM_Y_OFFSET, 0], q=_ARM_QUAT)
                )

            # Randomize ball positions in front of each arm
            for i in range(self.n_arms):
                arm_x = i * self.arm_spacing
                xyz = torch.zeros((b, 3), device=self.device)
                xyz[:, 0] = arm_x + (torch.rand((b,), device=self.device) * 0.2 - 0.1)
                xyz[:, 1] = torch.rand((b,), device=self.device) * 0.2 - 0.1
                xyz[:, 2] = self.ball_radius + torch.rand((b,), device=self.device) * 0.15
                qs = torch.zeros((b, 4), device=self.device)
                qs[:, 0] = 1.0
                self.balls[i].set_pose(Pose.create_from_pq(xyz, qs))

    def _agent(self, idx: int) -> Panda:
        return self.agent.agents[idx]

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------

    def evaluate(self):
        all_reached = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        info = {}
        for i in range(self.n_arms):
            tcp_pos = self._agent(i).tcp.pose.p
            ball_pos = self.balls[i].pose.p
            dist = torch.linalg.norm(tcp_pos - ball_pos, dim=1)
            reached = dist <= self.reach_thresh
            info[f"ball_{i}_reached"] = reached
            info[f"dist_{i}"] = dist
            all_reached = torch.logical_and(all_reached, reached)
        info["success"] = all_reached
        return info

    # ------------------------------------------------------------------
    # Observation extra — same layout as MultiPickBall for compatibility
    # ------------------------------------------------------------------

    def _get_obs_extra(self, info: dict):
        obs = {}
        if "state" in self.obs_mode:
            for i in range(self.n_arms):
                agent = self._agent(i)
                tcp_pos = agent.tcp.pose.p
                ball_pos = self.balls[i].pose.p
                tcp_to_ball = ball_pos - tcp_pos

                obs[f"agent_{i}_tcp_pos"] = tcp_pos                         # 3
                obs[f"agent_{i}_ball_pos"] = ball_pos                       # 3
                obs[f"agent_{i}_ball_vel"] = torch.zeros_like(ball_pos)     # 3 (static ball)
                obs[f"agent_{i}_goal_pos"] = ball_pos                       # 3 (goal = ball pos)
                obs[f"agent_{i}_tcp_to_ball"] = tcp_to_ball                 # 3
                obs[f"agent_{i}_ball_to_goal"] = torch.zeros_like(ball_pos) # 3 (zero)
        return obs

    # ------------------------------------------------------------------
    # Reward — DEAD SIMPLE: just distance
    # ------------------------------------------------------------------

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        total_reward = torch.zeros(self.num_envs, device=self.device)

        for i in range(self.n_arms):
            dist = info[f"dist_{i}"]
            # Smooth reach reward: 0 when far, ~1 when close
            reach = 1.0 - torch.tanh(5.0 * dist)
            total_reward = total_reward + reach

            # Bonus for reaching
            total_reward = total_reward + 5.0 * (dist < self.reach_thresh).float()

        # Normalize by n_arms
        return total_reward / self.n_arms

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        # Max reward per arm ≈ 6.0, so divide by 6
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 6.0
