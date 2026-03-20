"""
MultiPickBall — N independent Panda arms, each picks its own ball.

N Panda arms arranged along the x-axis at 0.4m spacing. Each arm has one
ball (sphere r=0.02) and one goal position in its own workspace. Agents are
fully independent with no shared objects or coordination requirements.

Observation layout (flat state tensor, total = N * 36):
    [agent_0_qpos(9), agent_0_qvel(9), ..., agent_{N-1}_qpos(9), agent_{N-1}_qvel(9),   <- N*18
     agent_0_extra(18), ..., agent_{N-1}_extra(18)]                                       <- N*18

Per-agent extra (18 dims):
    tcp_pos(3) + ball_pos(3) + ball_vel(3) + goal_pos(3) + tcp_to_ball(3) + ball_to_goal(3)

Reward: sum of per-agent dense rewards (reach + grasp + place).
Success: all balls placed at their respective goal positions.
"""

from typing import Any, List

import numpy as np
import sapien
import torch

from mani_skill.agents.multi_agent import MultiAgent
from mani_skill.agents.robots.panda import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig


# Register 2-agent variant (most common; add more variants as needed)
@register_env("MultiPickBall-v1", max_episode_steps=200)
class MultiPickBall(BaseEnv):
    """
    N Panda arms each independently pick their own ball and place it at a goal.
    """

    SUPPORTED_ROBOTS = [
        ("panda", "panda"),
        ("panda", "panda", "panda"),
        ("panda", "panda", "panda", "panda"),
    ]

    # configurable
    n_arms: int = 2                # number of robot arms / agents
    arm_spacing: float = 0.4       # x-axis spacing between arms (m)
    ball_radius: float = 0.02
    goal_thresh: float = 0.05      # success radius for ball-at-goal
    robot_init_qpos_noise: float = 0.02

    def __init__(
        self,
        *args,
        n_arms: int = 2,
        robot_uids=None,
        robot_init_qpos_noise: float = 0.02,
        **kwargs,
    ):
        self.n_arms = n_arms
        self.robot_init_qpos_noise = robot_init_qpos_noise
        if robot_uids is None:
            robot_uids = tuple(["panda"] * n_arms)
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    # ------------------------------------------------------------------
    # Simulation / sensor config
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
        # Place arms along x-axis, all facing +y
        poses = [
            sapien.Pose(p=[i * self.arm_spacing, 0, 0])
            for i in range(self.n_arms)
        ]
        super()._load_agent(options, poses)

    def _load_scene(self, options: dict):
        # Flat ground plane
        floor_width = 500 if self.scene.parallel_in_single_scene else 20
        build_ground(self.scene, floor_width=floor_width, altitude=0)

        self.balls: List[Any] = []
        self.goal_sites: List[Any] = []
        for i in range(self.n_arms):
            ball = actors.build_sphere(
                self.scene,
                radius=self.ball_radius,
                color=[1.0, 0.3 + 0.1 * i, 0.0, 1.0],
                name=f"ball_{i}",
                initial_pose=sapien.Pose(p=[i * self.arm_spacing, 0.3, self.ball_radius]),
            )
            self.balls.append(ball)

            goal = actors.build_sphere(
                self.scene,
                radius=self.goal_thresh,
                color=[0.0, 1.0, 0.3, 0.4],
                name=f"goal_{i}",
                body_type="kinematic",
                add_collision=False,
                initial_pose=sapien.Pose(p=[i * self.arm_spacing, 0.3, 0.15]),
            )
            self._hidden_objects.append(goal)
            self.goal_sites.append(goal)

    # ------------------------------------------------------------------
    # Episode initialisation
    # ------------------------------------------------------------------

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)

            # Panda rest qpos (7 arm joints + 2 gripper fingers open)
            rest_qpos = np.array(
                [0.0, np.pi / 8, 0, -np.pi * 5 / 8, 0, np.pi * 3 / 4, np.pi / 4, 0.04, 0.04]
            )

            for i, sub_agent in enumerate(self.agent.agents):
                # Add small noise to arm joints only
                qpos = (
                    self._episode_rng.normal(
                        0, self.robot_init_qpos_noise, (b, len(rest_qpos))
                    )
                    + rest_qpos
                )
                qpos[:, 7:] = 0.04  # keep gripper fully open
                sub_agent.reset(qpos)
                # Set base pose (facing +y along x-axis)
                sub_agent.robot.set_pose(
                    sapien.Pose(p=[i * self.arm_spacing, 0, 0])
                )

            for i in range(self.n_arms):
                arm_x = i * self.arm_spacing
                # Ball: random position in arm's workspace
                # y: 0.2 ~ 0.45 m in front, x: ±0.12 m around arm base
                xyz = torch.zeros((b, 3), device=self.device)
                xyz[:, 0] = arm_x + (torch.rand((b,), device=self.device) * 0.24 - 0.12)
                xyz[:, 1] = 0.20 + torch.rand((b,), device=self.device) * 0.25
                xyz[:, 2] = self.ball_radius
                # sphere — orientation doesn't matter, use identity quaternion (w,x,y,z)
                qs = torch.zeros((b, 4), device=self.device)
                qs[:, 0] = 1.0  # w=1 → identity
                self.balls[i].set_pose(Pose.create_from_pq(xyz, qs))

                # Goal: same x-range but elevated ~10 cm above ground
                goal_xyz = torch.zeros((b, 3), device=self.device)
                goal_xyz[:, 0] = arm_x + (torch.rand((b,), device=self.device) * 0.24 - 0.12)
                goal_xyz[:, 1] = 0.20 + torch.rand((b,), device=self.device) * 0.25
                goal_xyz[:, 2] = 0.10 + torch.rand((b,), device=self.device) * 0.10
                self.goal_sites[i].set_pose(Pose.create_from_pq(goal_xyz))

    # ------------------------------------------------------------------
    # Agent properties
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
            dist = torch.linalg.norm(
                self.goal_sites[i].pose.p - self.balls[i].pose.p, dim=1
            )
            placed = dist <= self.goal_thresh
            info[f"ball_{i}_placed"] = placed
            all_placed = torch.logical_and(all_placed, placed)
        info["success"] = all_placed
        return info

    # ------------------------------------------------------------------
    # Observation extra
    # ------------------------------------------------------------------

    def _get_obs_extra(self, info: dict):
        """
        Return per-agent extras in a flat dict so ManiSkill concatenates them
        in insertion order after the proprioception block.

        Obs layout extra block = [agent_0_extra(18), agent_1_extra(18), ...]
        Each extra_i = tcp_pos(3)+ball_pos(3)+ball_vel(3)+goal_pos(3)+tcp_to_ball(3)+ball_to_goal(3)
        """
        obs = {}
        if "state" in self.obs_mode:
            for i in range(self.n_arms):
                agent = self._agent(i)
                tcp_pos = agent.tcp.pose.p                              # (B, 3)
                ball_pos = self.balls[i].pose.p                        # (B, 3)
                ball_vel = self.balls[i].linear_velocity               # (B, 3)
                goal_pos = self.goal_sites[i].pose.p                   # (B, 3)
                tcp_to_ball = ball_pos - tcp_pos                       # (B, 3)
                ball_to_goal = goal_pos - ball_pos                     # (B, 3)

                obs[f"agent_{i}_tcp_pos"] = tcp_pos
                obs[f"agent_{i}_ball_pos"] = ball_pos
                obs[f"agent_{i}_ball_vel"] = ball_vel
                obs[f"agent_{i}_goal_pos"] = goal_pos
                obs[f"agent_{i}_tcp_to_ball"] = tcp_to_ball
                obs[f"agent_{i}_ball_to_goal"] = ball_to_goal
        return obs

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        total_reward = torch.zeros(self.num_envs, device=self.device)

        for i in range(self.n_arms):
            agent = self._agent(i)
            tcp_pos = agent.tcp.pose.p
            ball_pos = self.balls[i].pose.p
            goal_pos = self.goal_sites[i].pose.p

            # --- Stage 1: reach the ball ---
            tcp_to_ball = torch.linalg.norm(ball_pos - tcp_pos, dim=1)
            reach_reward = 1.0 - torch.tanh(5.0 * tcp_to_ball)

            # --- Stage 2: grasp ---
            is_grasped = agent.is_grasping(self.balls[i])
            grasp_reward = reach_reward + 2.0 * is_grasped.float()

            # Tip height symmetry for good grasp posture
            tip1_z = agent.finger1_link.pose.p[:, 2]
            tip2_z = agent.finger2_link.pose.p[:, 2]
            tip_sym = 1.0 - torch.tanh(5.0 * torch.abs(tip1_z - tip2_z))
            grasp_reward = grasp_reward + tip_sym

            # --- Stage 3: place at goal (only once grasped) ---
            ball_to_goal = torch.linalg.norm(goal_pos - ball_pos, dim=1)
            place_reward = 1.0 - torch.tanh(5.0 * ball_to_goal)

            # Staged: place reward only counts when grasped
            agent_reward = grasp_reward
            agent_reward[is_grasped] = 5.0 + 3.0 * place_reward[is_grasped]

            # Success bonus
            agent_reward[info[f"ball_{i}_placed"]] = 10.0

            total_reward = total_reward + agent_reward

        # Normalize by n_arms so reward scale is independent of N
        return total_reward / self.n_arms

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ):
        # Max per-agent reward ≈ 10, total ≈ 10 after normalization
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 10.0
