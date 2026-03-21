"""
MultiPushCube-v1 — Multi-agent PushCube using TableSceneBuilder.

Two Panda arms on opposite sides of a table (placed by TableSceneBuilder),
each pushes its own cube to a goal. Reward identical to official PushCube.

TableSceneBuilder coordinate system:
  z = 0 is the TABLE SURFACE (not the ground).
  Ground is at z = -0.92m.
  For ("panda","panda"), arms are at y = -0.75 and y = +0.75, facing each other.

This is a direct multi-agent copy of official PushCube-v1.
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
    """Two Panda arms on a table, each pushes its own cube to a goal."""

    SUPPORTED_ROBOTS = [("panda", "panda")]

    n_arms: int = 2
    cube_half_size: float = 0.02
    goal_radius: float = 0.1       # same as official PushCube
    robot_init_qpos_noise: float = 0.02

    def __init__(self, *args, robot_uids=("panda", "panda"),
                 robot_init_qpos_noise: float = 0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
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
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    # Do NOT override _load_agent — let BaseEnv + TableSceneBuilder handle it

    def _load_scene(self, options: dict):
        # TableSceneBuilder creates the table and handles arm placement
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        self.cubes: List[Any] = []
        self.goal_sites: List[Any] = []

        for i in range(self.n_arms):
            cube = actors.build_cube(
                self.scene,
                half_size=self.cube_half_size,
                color=[1, 0, 0, 1],
                name=f"cube_{i}",
                body_type="dynamic",
                # z=0 is table surface, cube sits on top
                initial_pose=sapien.Pose(p=[0, 0, self.cube_half_size]),
            )
            self.cubes.append(cube)

            goal = actors.build_red_white_target(
                self.scene,
                radius=self.goal_radius,
                thickness=1e-5,
                name=f"goal_{i}",
                add_collision=False,
                body_type="kinematic",
                initial_pose=sapien.Pose(p=[0, 0, 1e-3], q=euler2quat(0, np.pi / 2, 0)),
            )
            self.goal_sites.append(goal)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)

            # Let TableSceneBuilder handle ALL robot init (qpos, pose)
            self.table_scene.initialize(env_idx)

            # Arms are now at:
            #   agent 0: y = -0.75, facing +y (toward center)
            #   agent 1: y = +0.75, facing -y (toward center)
            # Table surface is z = 0.
            # Each arm's workspace is roughly x ∈ [-0.2, 0.2], y from its side toward center.

            # Cube 0: in agent 0's workspace (y < 0 side)
            cube0_xyz = torch.zeros((b, 3), device=self.device)
            cube0_xyz[:, 0] = torch.rand((b,), device=self.device) * 0.2 - 0.1  # x ∈ [-0.1, 0.1]
            cube0_xyz[:, 1] = -0.3 + torch.rand((b,), device=self.device) * 0.2  # y ∈ [-0.3, -0.1]
            cube0_xyz[:, 2] = self.cube_half_size
            qs = torch.zeros((b, 4), device=self.device)
            qs[:, 0] = 1.0
            self.cubes[0].set_pose(Pose.create_from_pq(cube0_xyz, qs))
            self.cubes[0].set_linear_velocity(torch.zeros((b, 3), device=self.device))
            self.cubes[0].set_angular_velocity(torch.zeros((b, 3), device=self.device))

            # Goal 0: further from agent 0 (closer to center)
            goal0_xyz = cube0_xyz.clone()
            goal0_xyz[:, 1] = goal0_xyz[:, 1] + 0.1 + self.goal_radius  # push toward center
            goal0_xyz[:, 2] = 1e-3
            self.goal_sites[0].set_pose(
                Pose.create_from_pq(goal0_xyz, euler2quat(0, np.pi / 2, 0))
            )

            # Cube 1: in agent 1's workspace (y > 0 side)
            cube1_xyz = torch.zeros((b, 3), device=self.device)
            cube1_xyz[:, 0] = torch.rand((b,), device=self.device) * 0.2 - 0.1
            cube1_xyz[:, 1] = 0.1 + torch.rand((b,), device=self.device) * 0.2  # y ∈ [0.1, 0.3]
            cube1_xyz[:, 2] = self.cube_half_size
            self.cubes[1].set_pose(Pose.create_from_pq(cube1_xyz, qs))
            self.cubes[1].set_linear_velocity(torch.zeros((b, 3), device=self.device))
            self.cubes[1].set_angular_velocity(torch.zeros((b, 3), device=self.device))

            # Goal 1: further from agent 1 (closer to center)
            goal1_xyz = cube1_xyz.clone()
            goal1_xyz[:, 1] = goal1_xyz[:, 1] - 0.1 - self.goal_radius  # push toward center
            goal1_xyz[:, 2] = 1e-3
            self.goal_sites[1].set_pose(
                Pose.create_from_pq(goal1_xyz, euler2quat(0, np.pi / 2, 0))
            )

    def _agent(self, idx: int) -> Panda:
        return self.agent.agents[idx]

    # ------------------------------------------------------------------
    # Evaluate
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
    # Obs extra — 18 dims per agent
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
    # Reward — same staged logic as official PushCube-v1
    # ------------------------------------------------------------------

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        total_reward = torch.zeros(self.num_envs, device=self.device)

        for i in range(self.n_arms):
            agent = self._agent(i)
            tcp_pos = agent.tcp.pose.p
            cube_pos = self.cubes[i].pose.p
            goal_pos = self.goal_sites[i].pose.p

            # Stage 1: reach the push position (behind cube, away from goal)
            # Compute direction from cube to goal
            direction = goal_pos[:, :2] - cube_pos[:, :2]
            direction_norm = torch.linalg.norm(direction, dim=1, keepdim=True).clamp(min=1e-6)
            direction = direction / direction_norm

            push_pos = cube_pos.clone()
            push_pos[:, :2] = push_pos[:, :2] - direction * (self.cube_half_size + 0.005)

            tcp_to_push_dist = torch.linalg.norm(tcp_pos - push_pos, dim=1)
            reaching_reward = 1.0 - torch.tanh(5.0 * tcp_to_push_dist)
            agent_reward = reaching_reward

            # Stage 2: push cube toward goal (once reached)
            reached = tcp_to_push_dist < 0.01
            obj_to_goal_dist = info[f"cube_{i}_to_goal_dist"]
            place_reward = 1.0 - torch.tanh(5.0 * obj_to_goal_dist)
            agent_reward = agent_reward + place_reward * reached.float()

            # Stage 3: keep cube on table
            z_deviation = torch.abs(cube_pos[:, 2] - self.cube_half_size)
            z_reward = 1.0 - torch.tanh(5.0 * z_deviation)
            agent_reward = agent_reward + place_reward * z_reward * reached.float()

            # Success bonus
            agent_reward[info[f"cube_{i}_placed"]] = 4.0

            total_reward = total_reward + agent_reward

        return total_reward / self.n_arms

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 4.0