from typing import Any, Tuple

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

from mani_skill.agents.multi_agent import MultiAgent
from mani_skill.agents.robots.panda import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig


@register_env("TwoRobotPassStick-v1", max_episode_steps=150)
class TwoRobotPassStick(BaseEnv):
    """
    **Task Description:**
    A collaborative task where two robot arms must pass a stick from one to the other
    without dropping it. The left robot starts holding the stick and must hand it off
    to the right robot, which then lifts it to a goal height.

    If the stick falls onto the table (z < stick_radius + table_height + epsilon),
    the task is considered failed.

    **Randomizations:**
    - The stick's initial position is randomized within the left robot's reach
    - The goal height for the right robot is randomized

    **Success Conditions:**
    - The right robot is grasping the stick
    - The stick is at the goal height
    - The left robot has released the stick
    """

    SUPPORTED_ROBOTS = [("panda_wristcam", "panda_wristcam")]
    agent: MultiAgent[Tuple[Panda, Panda]]

    stick_radius = 0.015
    stick_half_length = 0.12
    goal_thresh = 0.03

    def __init__(
        self,
        *args,
        robot_uids=("panda_wristcam", "panda_wristcam"),
        robot_init_qpos_noise=0.02,
        **kwargs
    ):
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
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[0.6, 0.2, 0.4], target=[-0.1, 0, 0.1])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(
            options, [sapien.Pose(p=[0, -1, 0]), sapien.Pose(p=[0, 1, 0])]
        )

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # Build stick as a cylinder (lying horizontally along y-axis)
        self.stick = actors.build_cylinder(
            self.scene,
            radius=self.stick_radius,
            half_length=self.stick_half_length,
            color=[0.6, 0.3, 0.1, 1],  # brown
            name="stick",
            initial_pose=sapien.Pose(p=[0, -0.2, 0.1]),
        )

        # Goal marker (where the stick should end up)
        self.goal_site = actors.build_sphere(
            self.scene,
            radius=self.goal_thresh,
            color=[0, 1, 0, 1],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )
        self._hidden_objects.append(self.goal_site)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # Stick starts on the left side, slightly above table
            stick_xyz = torch.zeros((b, 3))
            stick_xyz[:, 0] = torch.rand((b,)) * 0.1 - 0.05
            stick_xyz[:, 1] = -0.15 - torch.rand((b,)) * 0.05
            stick_xyz[:, 2] = self.stick_radius + 0.005  # just above table

            # Stick lies along the y-axis by default (cylinder axis = z in sapien)
            # Rotate 90 degrees around x-axis so cylinder lies along y
            stick_q = torch.tensor(
                euler2quat(np.pi / 2, 0, 0), dtype=torch.float32
            ).unsqueeze(0).expand(b, -1)
            self.stick.set_pose(Pose.create_from_pq(p=stick_xyz, q=stick_q))

            # Goal is on the right side, elevated
            goal_xyz = torch.zeros((b, 3))
            goal_xyz[:, 0] = torch.rand((b,)) * 0.1 - 0.05
            goal_xyz[:, 1] = 0.15 + torch.rand((b,)) * 0.1 - 0.05
            goal_xyz[:, 2] = 0.15 + torch.rand((b,)) * 0.15  # 0.15 - 0.30 height
            self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))

            # Track if stick has ever been grasped, and if it was dropped after
            # Use full num_envs size so evaluate() works for all envs
            if not hasattr(self, "stick_ever_grasped") or self.stick_ever_grasped.shape[0] != self.num_envs:
                self.stick_ever_grasped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                self.stick_dropped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            # Reset only the envs being reinitialized
            self.stick_ever_grasped[env_idx] = False
            self.stick_dropped[env_idx] = False

    @property
    def left_agent(self) -> Panda:
        return self.agent.agents[0]

    @property
    def right_agent(self) -> Panda:
        return self.agent.agents[1]

    def evaluate(self):
        stick_pos = self.stick.pose.p
        goal_pos = self.goal_site.pose.p

        is_left_grasping = self.left_agent.is_grasping(self.stick)
        is_right_grasping = self.right_agent.is_grasping(self.stick)

        # Track if anyone has ever grasped the stick
        self.stick_ever_grasped = self.stick_ever_grasped | is_left_grasping | is_right_grasping

        # Only count as dropped if it was previously grasped then fell
        table_z = self.stick_radius + 0.002
        stick_on_table = stick_pos[:, 2] < table_z
        nobody_holding = ~is_left_grasping & ~is_right_grasping
        just_dropped = stick_on_table & nobody_holding & self.stick_ever_grasped
        self.stick_dropped = self.stick_dropped | just_dropped

        # Success: right robot holds stick at goal, left has released
        stick_at_goal = (
            torch.linalg.norm(stick_pos - goal_pos, axis=1) <= self.goal_thresh
        )
        success = stick_at_goal & is_right_grasping & ~is_left_grasping

        return {
            "success": success,
            "fail": self.stick_dropped,
            "is_left_grasping": is_left_grasping,
            "is_right_grasping": is_right_grasping,
            "stick_at_goal": stick_at_goal,
            "stick_dropped": self.stick_dropped,
        }

    def get_state_dict(self):
        state = super().get_state_dict()
        state["stick_ever_grasped"] = self.stick_ever_grasped
        state["stick_dropped"] = self.stick_dropped
        return state

    def set_state_dict(self, state_dict):
        super().set_state_dict(state_dict)
        self.stick_ever_grasped = state_dict["stick_ever_grasped"]
        self.stick_dropped = state_dict["stick_dropped"]

    def _get_obs_extra(self, info: dict):
        obs = dict(
            left_arm_tcp=self.left_agent.tcp.pose.raw_pose,
            right_arm_tcp=self.right_agent.tcp.pose.raw_pose,
        )
        if "state" in self.obs_mode:
            obs.update(
                stick_pose=self.stick.pose.raw_pose,
                goal_pos=self.goal_site.pose.p,
                left_tcp_to_stick=self.stick.pose.p - self.left_agent.tcp.pose.p,
                right_tcp_to_stick=self.stick.pose.p - self.right_agent.tcp.pose.p,
                stick_to_goal=self.goal_site.pose.p - self.stick.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        stick_pos = self.stick.pose.p
        goal_pos = self.goal_site.pose.p

        # If stick is dropped, no reward (agent is stuck)
        reward = torch.zeros(self.num_envs, device=self.device)
        not_dropped = ~info["stick_dropped"]

        # Stage 1: Left robot reaches and grasps the stick
        left_tcp_to_stick = torch.linalg.norm(
            self.left_agent.tcp.pose.p - stick_pos, axis=1
        )
        reach_reward_left = 1 - torch.tanh(5 * left_tcp_to_stick)
        stage_1_reward = reach_reward_left + info["is_left_grasping"].float()
        reward[not_dropped] = stage_1_reward[not_dropped] / 2

        left_grasped = info["is_left_grasping"] & not_dropped

        # Stage 2: Left robot brings stick toward the middle (handover zone y~0)
        stick_to_middle = torch.abs(stick_pos[:, 1])  # distance from y=0
        handover_reward = 1 - torch.tanh(5 * stick_to_middle)
        # Right robot reaches toward the stick
        right_tcp_to_stick = torch.linalg.norm(
            self.right_agent.tcp.pose.p - stick_pos, axis=1
        )
        reach_reward_right = 1 - torch.tanh(5 * right_tcp_to_stick)
        stage_2_reward = handover_reward + reach_reward_right
        reward[left_grasped] = 2 + stage_2_reward[left_grasped] / 2

        # Stage 3: Both robots grasping (handover moment)
        both_grasping = left_grasped & info["is_right_grasping"]
        # Reward left robot for releasing (opening gripper)
        gripper_width = (self.left_agent.robot.get_qlimits()[0, -1, 1] * 2).to(
            self.device
        )
        left_openness = (
            torch.sum(self.left_agent.robot.get_qpos()[:, -2:], axis=1) / gripper_width
        )
        stage_3_reward = left_openness
        reward[both_grasping] = 5 + stage_3_reward[both_grasping]

        # Stage 4: Right robot has stick, left has released -> bring to goal
        right_only = info["is_right_grasping"] & ~info["is_left_grasping"] & not_dropped
        stick_to_goal = torch.linalg.norm(stick_pos - goal_pos, axis=1)
        place_reward = 1 - torch.tanh(5 * stick_to_goal)
        # Left arm returns to rest
        left_arm_leave_reward = 1 - torch.tanh(
            5 * (self.left_agent.tcp.pose.p[:, 1] + 0.2).abs()
        )
        stage_4_reward = 2 * place_reward + left_arm_leave_reward
        reward[right_only] = 7 + stage_4_reward[right_only]

        reward[info["success"]] = 12

        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 12
