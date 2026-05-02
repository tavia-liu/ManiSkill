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
    A collaborative task where two robot arms must pass a stick from one to the other.
    The left robot starts near the stick and must hand it off to the right robot.

    **Randomizations:**
    - The stick's initial position is randomized within the left robot's reach

    **Success Conditions:**
    - The right robot is grasping the stick
    - The left robot has released the stick
    - The stick is on the right side of the table and lifted off the table surface
    """

    SUPPORTED_ROBOTS = [("panda_wristcam", "panda_wristcam")]
    agent: MultiAgent[Tuple[Panda, Panda]]

    stick_radius = 0.015
    stick_half_length = 0.12
    handover_y = 0.10  # stick must end up past this y-coordinate
    handover_min_z = 0.10  # and lifted at least this high

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

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            self.left_init_qpos = self.left_agent.robot.get_qpos()

            # Stick starts on the left side, slightly above table
            stick_xyz = torch.zeros((b, 3))
            stick_xyz[:, 0] = torch.rand((b,)) * 0.1 - 0.05
            stick_xyz[:, 1] = -0.15 - torch.rand((b,)) * 0.05
            stick_xyz[:, 2] = self.stick_radius + 0.005  # just above table

            # Cylinder default axis is z; rotate 90° about x so it lies along y.
            stick_q = torch.tensor(
                euler2quat(np.pi / 2, 0, 0),
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0).expand(b, -1)
            self.stick.set_pose(Pose.create_from_pq(p=stick_xyz, q=stick_q))

    @property
    def left_agent(self) -> Panda:
        return self.agent.agents[0]

    @property
    def right_agent(self) -> Panda:
        return self.agent.agents[1]

    def evaluate(self):
        stick_pos = self.stick.pose.p

        is_left_grasping = self.left_agent.is_grasping(self.stick)
        is_right_grasping = self.right_agent.is_grasping(self.stick)

        stick_passed = (stick_pos[:, 1] >= self.handover_y) & (
            stick_pos[:, 2] >= self.handover_min_z
        ) # stick has crossed the midline and is lifted off the table             
        success = stick_passed & is_right_grasping & ~is_left_grasping

        return {
            "success": success,
            "is_left_grasping": is_left_grasping,
            "is_right_grasping": is_right_grasping,
            "stick_passed": stick_passed,
        }

    def _get_obs_extra(self, info: dict):
        obs = dict(
            left_arm_tcp=self.left_agent.tcp.pose.raw_pose,
            right_arm_tcp=self.right_agent.tcp.pose.raw_pose,
        )
        if "state" in self.obs_mode:
            obs.update(
                stick_pose=self.stick.pose.raw_pose,
                left_tcp_to_stick=self.stick.pose.p - self.left_agent.tcp.pose.p,
                right_tcp_to_stick=self.stick.pose.p - self.right_agent.tcp.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        stick_pos = self.stick.pose.p
        left_tcp = self.left_agent.tcp.pose.p
        right_tcp = self.right_agent.tcp.pose.p
        is_left_grasping = info["is_left_grasping"]
        is_right_grasping = info["is_right_grasping"]

        # Stage 1: left reaches stick and pushes/lifts it past the midline (y=0).
        # One-sided clipped reward: saturates once across, no overshoot incentive.
        left_tcp_to_stick = torch.linalg.norm(left_tcp - stick_pos, axis=1)
        reach_left = 1 - torch.tanh(5 * left_tcp_to_stick)
        stick_to_other_side = 1 - torch.tanh(
            5 * torch.clamp(-stick_pos[:, 1], min=0.0)
        )
        reward = (reach_left + stick_to_other_side) / 2

        stick_at_other_side = stick_pos[:, 1] >= 0.0

        # Stage 2: right arm approaches with pre-shaped gripper, then grasps.
        # Left arm only starts retreating after right has secured the grasp.
        right_tcp_to_stick = torch.linalg.norm(right_tcp - stick_pos, axis=1)
        right_reach = 1 - torch.tanh(5 * right_tcp_to_stick)

        # Pre-shape only matters during approach (within 0.15m of stick)
        right_tip1 = self.right_agent.finger1_link.pose.p
        right_tip2 = self.right_agent.finger2_link.pose.p
        tip_height = 1 - torch.tanh(
            5 * torch.abs(right_tip1[:, 2] - right_tip2[:, 2])
        )
        tip_width = 1 - torch.tanh(
            5 * torch.abs(
                torch.linalg.norm(right_tip1 - right_tip2, axis=1) - 0.07
            )
        )
        right_approaching = right_tcp_to_stick < 0.15
        tip_reward = (tip_height + tip_width) / 2 * right_approaching.float()

        left_arm_leave = 1 - torch.tanh(5 * (left_tcp[:, 1] + 0.2).abs())
        left_arm_leave = left_arm_leave * is_right_grasping.float()

        stage_2_reward = (
            right_reach + tip_reward + left_arm_leave + 2 * is_right_grasping.float()
        )
        reward[stick_at_other_side] = 2 + stage_2_reward[stick_at_other_side]

        # Stage 3: right grasping & stick on right side -> lift to handover thresholds,
        # left lets go and retreats home.
        stage_3_active = is_right_grasping & stick_at_other_side

        lift_y_reward = 1 - torch.tanh(
            5 * torch.clamp(self.handover_y - stick_pos[:, 1], min=0.0)
        )
        lift_z_reward = 1 - torch.tanh(
            5 * torch.clamp(self.handover_min_z - stick_pos[:, 2], min=0.0)
        )

        # Ungrasp reward: smooth gradient as gripper opens, saturates at 1.0
        # the moment is_left_grasping flips false (stack_cube pattern).
        gripper_width = (self.left_agent.robot.get_qlimits()[0, -1, 1] * 2).to(
            self.device
        )
        ungrasp_reward_left = (
            torch.sum(self.left_agent.robot.get_qpos()[:, -2:], axis=1) / gripper_width
        )
        ungrasp_reward_left[~is_left_grasping] = 1.0

        left_arm_retreat = 1 - torch.tanh(
            torch.linalg.norm(
                self.left_agent.robot.get_qpos()[:, :-2]
                - self.left_init_qpos[:, :-2],
                axis=1,
            )
        )

        stage_3_reward = (
            lift_y_reward + lift_z_reward + ungrasp_reward_left + left_arm_retreat
        )
        reward[stage_3_active] = 8 + stage_3_reward[stage_3_active]

        reward[info["success"]] = 14

        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 14
