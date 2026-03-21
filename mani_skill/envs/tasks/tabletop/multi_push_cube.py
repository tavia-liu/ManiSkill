"""
MultiPushCube-v1 — Multi-agent version of ManiSkill's PushCube-v1.

Uses the EXACT same scene setup as official PushCube (TableSceneBuilder),
so arm workspace and cube positions are correct. Each arm independently
pushes its own cube to a goal. Reward function copied from official PushCube.

This task is the multi-agent equivalent of official PushCube-v1.
If official PPO solves PushCube in 1.3M steps, HAPPO should solve this
in similar time since the agents are fully independent.
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
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig


@register_env("MultiPushCube-v1", max_episode_steps=50)
class MultiPushCube(BaseEnv):
    """N Panda arms on tables, each pushes a cube to a goal. Same as official PushCube."""

    SUPPORTED_ROBOTS = [
        ("panda", "panda"),
        ("panda", "panda", "panda"),
        ("panda", "panda", "panda", "panda"),
    ]

    n_arms: int = 2
    arm_spacing: float = 1.5       # enough space between tables
    cube_half_size: float = 0.02
    goal_radius: float = 0.1       # same as official PushCube
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
        pose = sapien_utils.look_at([3.0, 0, 2.0], [0.0, 0.0, 0.3])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        # Place each arm at its own table position
        # Official PushCube puts the arm at (-0.615, 0, 0) via TableSceneBuilder
        poses = [
            sapien.Pose(p=[i * self.arm_spacing - 0.615, 0, 0])
            for i in range(self.n_arms)
        ]
        super()._load_agent(options, poses)

    def _load_scene(self, options: dict):
        # Use TableSceneBuilder for proper table + arm setup
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        self.cubes: List[Any] = []
        self.goal_sites: List[Any] = []

        for i in range(self.n_arms):
            offset_x = i * self.arm_spacing

            cube = actors.build_cube(
                self.scene,
                half_size=self.cube_half_size,
                color=[1, 0, 0, 1],
                name=f"cube_{i}",
                body_type="dynamic",
                initial_pose=sapien.Pose(p=[offset_x, 0, self.cube_half_size]),
            )
            self.cubes.append(cube)

            goal = actors.build_red_white_target(
                self.scene,
                radius=self.goal_radius,
                thickness=1e-5,
                name=f"goal_{i}",
                add_collision=False,
                body_type="kinematic",
                initial_pose=sapien.Pose(
                    p=[offset_x + 0.1 + self.goal_radius, 0, 1e-3],
                    q=euler2quat(0, np.pi / 2, 0),
                ),
            )
            self.goal_sites.append(goal)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)

            # Let TableSceneBuilder handle robot init (adds qpos noise)
            self.table_scene.initialize(env_idx)

            for i in range(self.n_arms):
                offset_x = i * self.arm_spacing

                # Cube: random xy on table, same as official PushCube
                cube_xyz = torch.zeros((b, 3), device=self.device)
                cube_xyz[:, 0] = offset_x + (torch.rand((b,), device=self.device) * 0.2 - 0.1)
                cube_xyz[:, 1] = torch.rand((b,), device=self.device) * 0.2 - 0.1
                cube_xyz[:, 2] = self.cube_half_size
                qs = torch.zeros((b, 4), device=self.device)
                qs[:, 0] = 1.0
                self.cubes[i].set_pose(Pose.create_from_pq(cube_xyz, qs))
                self.cubes[i].set_linear_velocity(torch.zeros((b, 3), device=self.device))
                self.cubes[i].set_angular_velocity(torch.zeros((b, 3), device=self.device))

                # Goal: in front of cube (same logic as official PushCube)
                goal_xyz = cube_xyz.clone()
                goal_xyz[:, 0] = goal_xyz[:, 0] + 0.1 + self.goal_radius
                goal_xyz[:, 2] = 1e-3
                self.goal_sites[i].set_pose(
                    Pose.create_from_pq(goal_xyz, euler2quat(0, np.pi / 2, 0))
                )

    def _agent(self, idx: int) -> Panda:
        return self.agent.agents[idx]

    # ------------------------------------------------------------------
    # Evaluate — same as official PushCube
    # ------------------------------------------------------------------

    def evaluate(self):
        all_placed = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        info = {}
        for i in range(self.n_arms):
            cube_xy = self.cubes[i].pose.p[:, :2]
            goal_xy = self.goal_sites[i].pose.p[:, :2]
            dist = torch.linalg.norm(cube_xy - goal_xy, dim=1)
            cube_z = self.cubes[i].pose.p[:, 2]
            placed = (dist < self.goal_radius) & (cube_z < self.cube_half_size + 5e-3)
            info[f"cube_{i}_placed"] = placed
            info[f"cube_{i}_to_goal_dist"] = dist
            all_placed = torch.logical_and(all_placed, placed)
        info["success"] = all_placed
        return info

    # ------------------------------------------------------------------
    # Obs extra — 18 dims per agent for HARL compatibility
    # ------------------------------------------------------------------

    def _get_obs_extra(self, info: dict):
        obs = {}
        if "state" in self.obs_mode:
            for i in range(self.n_arms):
                agent = self._agent(i)
                tcp_pos = agent.tcp.pose.p
                cube_pos = self.cubes[i].pose.p
                cube_vel = self.cubes[i].linear_velocity
                goal_pos = self.goal_sites[i].pose.p
                tcp_to_cube = cube_pos - tcp_pos
                cube_to_goal = goal_pos - cube_pos

                obs[f"agent_{i}_tcp_pos"] = tcp_pos
                obs[f"agent_{i}_ball_pos"] = cube_pos
                obs[f"agent_{i}_ball_vel"] = cube_vel
                obs[f"agent_{i}_goal_pos"] = goal_pos
                obs[f"agent_{i}_tcp_to_ball"] = tcp_to_cube
                obs[f"agent_{i}_ball_to_goal"] = cube_to_goal
        return obs

    # ------------------------------------------------------------------
    # Reward — EXACT same logic as official PushCube-v1
    # ------------------------------------------------------------------

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        total_reward = torch.zeros(self.num_envs, device=self.device)

        for i in range(self.n_arms):
            agent = self._agent(i)
            tcp_pos = agent.tcp.pose.p
            cube_pos = self.cubes[i].pose.p
            goal_pos = self.goal_sites[i].pose.p

            # Stage 1: reach behind cube (push from behind)
            tcp_push_pose = cube_pos.clone()
            tcp_push_pose[:, 0] = tcp_push_pose[:, 0] - self.cube_half_size - 0.005
            tcp_to_push_dist = torch.linalg.norm(tcp_push_pose - tcp_pos, dim=1)
            reaching_reward = 1.0 - torch.tanh(5.0 * tcp_to_push_dist)
            agent_reward = reaching_reward

            # Stage 2: push cube toward goal (only when reached)
            reached = tcp_to_push_dist < 0.01
            obj_to_goal_dist = torch.linalg.norm(
                cube_pos[:, :2] - goal_pos[:, :2], dim=1
            )
            place_reward = 1.0 - torch.tanh(5.0 * obj_to_goal_dist)
            agent_reward = agent_reward + place_reward * reached.float()

            # Stage 3: keep cube on surface
            z_deviation = torch.abs(cube_pos[:, 2] - self.cube_half_size)
            z_reward = 1.0 - torch.tanh(5.0 * z_deviation)
            agent_reward = agent_reward + place_reward * z_reward * reached.float()

            # Success bonus
            agent_reward[info[f"cube_{i}_placed"]] = 4.0

            total_reward = total_reward + agent_reward

        return total_reward / self.n_arms

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 4.0