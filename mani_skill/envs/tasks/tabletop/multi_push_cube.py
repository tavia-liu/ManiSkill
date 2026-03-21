"""
MultiPushCube-v1 — Second sanity-check task for HARL + ManiSkill3 pipeline.

N Panda arms, each pushes its own cube to a nearby goal region on the ground.
No grasping or lifting required — just push the cube along the table surface.

Difficulty ladder:
  MultiReachBall  (trivial)  → reach only, no contact
  MultiPushCube   (easy)     → push, needs contact but no grasp  ← THIS
  MultiPickBall   (hard)     → pick and place, needs grasp + lift

Observation layout (flat state tensor, total = N * 36):
    [agent_0_qpos(9), agent_0_qvel(9), ..., agent_{N-1}_qpos(9), agent_{N-1}_qvel(9),
     agent_0_extra(18), ..., agent_{N-1}_extra(18)]

Per-agent extra (18 dims):
    tcp_pos(3) + cube_pos(3) + cube_vel(3) + goal_pos(3) + tcp_to_cube(3) + cube_to_goal(3)

Reward (per agent, staged):
    Stage 1: reach behind the cube              → 1 - tanh(5 * dist_to_push_pose)
    Stage 2: push cube toward goal (once reached) → 1 - tanh(5 * cube_to_goal_dist)
    Stage 3: keep cube flat on ground            → 1 - tanh(5 * z_deviation)
    Bonus: cube at goal                          → 4.0

Success: all cubes within goal_radius (0.05m) of their respective goals.
Episode length: 100 steps.
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


@register_env("MultiPushCube-v1", max_episode_steps=100)
class MultiPushCube(BaseEnv):
    """N Panda arms each push their own cube to a goal. No grasping required."""

    SUPPORTED_ROBOTS = [
        ("panda", "panda"),
        ("panda", "panda", "panda"),
        ("panda", "panda", "panda", "panda"),
    ]

    n_arms: int = 2
    arm_spacing: float = 0.4
    cube_half_size: float = 0.02
    goal_radius: float = 0.05       # success: cube within 5cm of goal
    robot_init_qpos_noise: float = 0.02

    def __init__(self, *args, n_arms: int = 2, robot_uids=None,
                 robot_init_qpos_noise: float = 0.02, **kwargs):
        self.n_arms = n_arms
        self.robot_init_qpos_noise = robot_init_qpos_noise
        if robot_uids is None:
            robot_uids = tuple(["panda"] * n_arms)
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    # ------------------------------------------------------------------
    # Sim config
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Scene loading
    # ------------------------------------------------------------------

    def _load_agent(self, options: dict):
        poses = [
            sapien.Pose(p=[i * self.arm_spacing, _ARM_Y_OFFSET, 0], q=_ARM_QUAT)
            for i in range(self.n_arms)
        ]
        super()._load_agent(options, poses)

    def _load_scene(self, options: dict):
        floor_width = 500 if self.scene.parallel_in_single_scene else 20
        build_ground(self.scene, floor_width=floor_width, altitude=0)

        self.cubes: List[Any] = []
        self.goal_sites: List[Any] = []

        for i in range(self.n_arms):
            # Dynamic cube — can be pushed by the robot
            cube = actors.build_cube(
                self.scene,
                half_size=self.cube_half_size,
                color=[12 / 255, 42 / 255, 160 / 255, 1.0],
                name=f"cube_{i}",
                body_type="dynamic",
                initial_pose=sapien.Pose(p=[i * self.arm_spacing, 0, self.cube_half_size]),
            )
            self.cubes.append(cube)

            # Visual-only goal marker (red/white target on the ground)
            goal = actors.build_red_white_target(
                self.scene,
                radius=self.goal_radius,
                thickness=1e-5,
                name=f"goal_{i}",
                add_collision=False,
                body_type="kinematic",
                initial_pose=sapien.Pose(
                    p=[i * self.arm_spacing, 0.15, 1e-3],
                    q=euler2quat(0, np.pi / 2, 0),
                ),
            )
            self.goal_sites.append(goal)

    # ------------------------------------------------------------------
    # Episode initialization
    # ------------------------------------------------------------------

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
                qpos[:, 7:] = 0.04  # gripper open
                sub_agent.reset(qpos)
                sub_agent.robot.set_pose(
                    sapien.Pose(p=[i * self.arm_spacing, _ARM_Y_OFFSET, 0], q=_ARM_QUAT)
                )

            for i in range(self.n_arms):
                arm_x = i * self.arm_spacing

                # Cube: random position in front of arm
                # After rotation, arm faces +y, so cube is along y-axis
                cube_xyz = torch.zeros((b, 3), device=self.device)
                cube_xyz[:, 0] = arm_x + (torch.rand((b,), device=self.device) * 0.1 - 0.05)
                cube_xyz[:, 1] = -0.1 + torch.rand((b,), device=self.device) * 0.1  # y in [-0.1, 0]
                cube_xyz[:, 2] = self.cube_half_size
                qs = torch.zeros((b, 4), device=self.device)
                qs[:, 0] = 1.0
                self.cubes[i].set_pose(Pose.create_from_pq(cube_xyz, qs))
                # Zero out velocity from previous episode
                self.cubes[i].set_linear_velocity(torch.zeros((b, 3), device=self.device))
                self.cubes[i].set_angular_velocity(torch.zeros((b, 3), device=self.device))

                # Goal: in front of cube (further along +y), on the ground
                goal_xyz = torch.zeros((b, 3), device=self.device)
                goal_xyz[:, 0] = arm_x + (torch.rand((b,), device=self.device) * 0.1 - 0.05)
                goal_xyz[:, 1] = 0.05 + torch.rand((b,), device=self.device) * 0.15  # y in [0.05, 0.2]
                goal_xyz[:, 2] = 1e-3
                self.goal_sites[i].set_pose(
                    Pose.create_from_pq(goal_xyz, euler2quat(0, np.pi / 2, 0))
                )

    # ------------------------------------------------------------------
    # Agent access
    # ------------------------------------------------------------------

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
            # Cube must be near goal AND still on the ground (not flipped off)
            placed = (dist < self.goal_radius) & (cube_z < self.cube_half_size + 0.005)
            info[f"cube_{i}_placed"] = placed
            info[f"cube_{i}_to_goal_dist"] = dist
            all_placed = torch.logical_and(all_placed, placed)
        info["success"] = all_placed
        return info

    # ------------------------------------------------------------------
    # Observation extra — same 18-dim layout as MultiPickBall
    # ------------------------------------------------------------------

    def _get_obs_extra(self, info: dict):
        obs = {}
        if "state" in self.obs_mode:
            for i in range(self.n_arms):
                agent = self._agent(i)
                tcp_pos = agent.tcp.pose.p                              # (B, 3)
                cube_pos = self.cubes[i].pose.p                        # (B, 3)
                cube_vel = self.cubes[i].linear_velocity               # (B, 3)
                goal_pos = self.goal_sites[i].pose.p                   # (B, 3)
                tcp_to_cube = cube_pos - tcp_pos                       # (B, 3)
                cube_to_goal = goal_pos - cube_pos                     # (B, 3)

                obs[f"agent_{i}_tcp_pos"] = tcp_pos                     # 3
                obs[f"agent_{i}_ball_pos"] = cube_pos                   # 3  (named ball for layout compat)
                obs[f"agent_{i}_ball_vel"] = cube_vel                   # 3
                obs[f"agent_{i}_goal_pos"] = goal_pos                   # 3
                obs[f"agent_{i}_tcp_to_ball"] = tcp_to_cube             # 3
                obs[f"agent_{i}_ball_to_goal"] = cube_to_goal           # 3
        return obs

    # ------------------------------------------------------------------
    # Reward — staged push reward (adapted from PushCube-v1)
    # ------------------------------------------------------------------

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        total_reward = torch.zeros(self.num_envs, device=self.device)

        for i in range(self.n_arms):
            agent = self._agent(i)
            tcp_pos = agent.tcp.pose.p
            cube_pos = self.cubes[i].pose.p
            goal_pos = self.goal_sites[i].pose.p

            # --- Stage 1: reach behind the cube (push from -y side) ---
            # The push pose is behind the cube relative to the goal direction
            push_dir = goal_pos[:, :2] - cube_pos[:, :2]
            push_dir_norm = torch.linalg.norm(push_dir, dim=1, keepdim=True).clamp(min=1e-6)
            push_dir = push_dir / push_dir_norm  # unit vector from cube to goal

            # Push pose: slightly behind cube, opposite to goal direction
            push_pose_xy = cube_pos[:, :2] - push_dir * (self.cube_half_size + 0.005)
            push_pose = torch.zeros_like(cube_pos)
            push_pose[:, :2] = push_pose_xy
            push_pose[:, 2] = cube_pos[:, 2]  # same height as cube

            tcp_to_push = torch.linalg.norm(tcp_pos - push_pose, dim=1)
            reach_reward = 1.0 - torch.tanh(5.0 * tcp_to_push)

            # --- Stage 2: push cube toward goal (only once reached) ---
            reached = tcp_to_push < 0.02  # within 2cm of push pose
            cube_to_goal_dist = info[f"cube_{i}_to_goal_dist"]
            push_reward = 1.0 - torch.tanh(5.0 * cube_to_goal_dist)

            # --- Stage 3: keep cube on the ground ---
            z_deviation = torch.abs(cube_pos[:, 2] - self.cube_half_size)
            z_reward = 1.0 - torch.tanh(5.0 * z_deviation)

            # Combine: reach always, push+z only when reached
            agent_reward = reach_reward
            agent_reward = agent_reward + push_reward * reached.float()
            agent_reward = agent_reward + push_reward * z_reward * reached.float()

            # Success bonus
            agent_reward[info[f"cube_{i}_placed"]] = 4.0

            total_reward = total_reward + agent_reward

        # Normalize by n_arms
        return total_reward / self.n_arms

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 4.0