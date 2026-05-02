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
from mani_skill.utils.geometry.rotation_conversions import quaternion_apply
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig


@register_env("TwoRobotPassStickTwoTables-v1", max_episode_steps=150)
class TwoRobotPassStickTwoTables(BaseEnv):
    """
    **Task Description:**
    Two robot arms must pass a stick between them across a *gap* between two
    separate raised tables. Because the gap is wider than the stick is long,
    the only way to move the stick from the left table to the right table is
    to lift it through the air — pushing it over the edge will let it fall
    into the gap. The left arm grasps the stick (typically near its -y end so
    the +y end protrudes for the receiver), lifts it across the gap, the right
    arm regrasps the protruding end, the left arm releases, and the right arm
    places the stick at a goal position on the right table.

    **Randomizations:**
    - Stick xy on the left table is randomized.
    - Goal xy on the right table is randomized.

    **Success Conditions:**
    - The stick is at the goal pose on the right table (xy within threshold,
      z near right-table top).
    - Neither arm is grasping the stick.
    - The right arm is approximately static.
    """

    SUPPORTED_ROBOTS = [("panda_wristcam", "panda_wristcam")]
    agent: MultiAgent[Tuple[Panda, Panda]]

    # Stick geometry: total length 0.20, well shorter than the gap so it cannot
    # bridge the gap by being pushed across.
    stick_radius = 0.015
    stick_half_length = 0.10

    # Two-table layout. Tables are raised platforms sitting on top of the
    # default workspace surface (z = 0). The gap between them along y is
    # wider than the stick is long.
    platform_height = 0.08
    platform_half_size_x = 0.20
    platform_half_size_y = 0.18
    gap_half_width = 0.13  # gap spans y in [-0.13, +0.13] -> 0.26 wide > stick (0.20)

    # Success thresholds.
    goal_xy_thresh = 0.04
    goal_z_thresh = 0.03  # how close to platform top the stick center should be

    # Mid-air clearance threshold (stick must rise above this while crossing the gap).
    transit_min_z = platform_height + 0.06

    def __init__(
        self,
        *args,
        robot_uids=("panda_wristcam", "panda_wristcam"),
        robot_init_qpos_noise=0.02,
        **kwargs,
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
        pose = sapien_utils.look_at(eye=[0.4, 0, 0.7], target=[-0.1, 0, 0.15])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[0.7, 0.3, 0.5], target=[-0.1, 0, 0.15])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(
            options, [sapien.Pose(p=[0, -1, 0]), sapien.Pose(p=[0, 1, 0])]
        )

    def _load_scene(self, options: dict):
        # Underlying floor/workspace + robot anchoring is handled by the
        # standard table scene. We then add two raised platforms on top with
        # a gap between them so the stick can't be slid across.
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # Center y of each platform (left at negative y, right at positive y).
        self._left_platform_y = -(self.gap_half_width + self.platform_half_size_y)
        self._right_platform_y = self.gap_half_width + self.platform_half_size_y

        platform_half_sizes = [
            self.platform_half_size_x,
            self.platform_half_size_y,
            self.platform_height / 2,
        ]
        self.left_platform = actors.build_box(
            self.scene,
            half_sizes=platform_half_sizes,
            color=[0.7, 0.55, 0.35, 1.0],
            name="left_platform",
            body_type="kinematic",
            initial_pose=sapien.Pose(
                p=[0.0, self._left_platform_y, self.platform_height / 2]
            ),
        )
        self.right_platform = actors.build_box(
            self.scene,
            half_sizes=platform_half_sizes,
            color=[0.7, 0.55, 0.35, 1.0],
            name="right_platform",
            body_type="kinematic",
            initial_pose=sapien.Pose(
                p=[0.0, self._right_platform_y, self.platform_height / 2]
            ),
        )

        # Stick: a slender cylinder. Default cylinder axis is local +x, but
        # historically these tasks rotate it about x to lay it along y. We
        # follow the same convention used in two_robot_pass_stick.py:
        # euler2quat(pi/2, 0, 0) makes the stick lie along world y.
        self.stick = actors.build_cylinder(
            self.scene,
            radius=self.stick_radius,
            half_length=self.stick_half_length,
            color=[0.6, 0.3, 0.1, 1.0],
            name="stick",
            initial_pose=sapien.Pose(p=[0, self._left_platform_y, 0.3]),
        )

        # Goal marker on the right platform (visual only).
        self.goal_site = actors.build_sphere(
            self.scene,
            radius=self.goal_xy_thresh,
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
            self.left_init_qpos = self.left_agent.robot.get_qpos()

            # Place the stick lying along y on the left platform. Sample its
            # center within the platform interior, leaving margin so the
            # stick (length 0.20) stays fully on the platform.
            stick_xyz = torch.zeros((b, 3))
            stick_xyz[:, 0] = (torch.rand((b,)) - 0.5) * 0.10  # +-0.05 in x
            y_margin = self.platform_half_size_y - self.stick_half_length - 0.02
            stick_xyz[:, 1] = (
                self._left_platform_y
                + (torch.rand((b,)) - 0.5) * 2 * max(y_margin, 0.0)
            )
            stick_xyz[:, 2] = self.platform_height + self.stick_radius + 0.002

            stick_q = (
                torch.tensor(
                    euler2quat(np.pi / 2, 0, 0),
                    dtype=torch.float32,
                    device=self.device,
                )
                .unsqueeze(0)
                .expand(b, -1)
            )
            self.stick.set_pose(Pose.create_from_pq(p=stick_xyz, q=stick_q))

            # Goal: on the right platform, random xy within a safe interior.
            goal_xyz = torch.zeros((b, 3))
            goal_xyz[:, 0] = (torch.rand((b,)) - 0.5) * 0.10
            goal_xyz[:, 1] = (
                self._right_platform_y
                + (torch.rand((b,)) - 0.5) * 2 * max(y_margin, 0.0)
            )
            goal_xyz[:, 2] = self.platform_height + self.stick_radius + 0.002
            self.goal_site.set_pose(Pose.create_from_pq(p=goal_xyz))
            self._goal_xyz = goal_xyz

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

        # Stick has crossed to the right side and is at-or-above platform height
        # at some point during transit (current pose check is a proxy).
        stick_on_right_side = stick_pos[:, 1] >= self.gap_half_width
        stick_lifted = stick_pos[:, 2] >= self.transit_min_z

        # At the goal: xy within threshold of goal, z near platform top.
        goal_xy_dist = torch.linalg.norm(
            stick_pos[:, :2] - self._goal_xyz[:, :2], axis=1
        )
        z_err = torch.abs(stick_pos[:, 2] - self._goal_xyz[:, 2])
        stick_at_goal = (goal_xy_dist <= self.goal_xy_thresh) & (z_err <= self.goal_z_thresh)

        is_right_static = self.right_agent.is_static(0.2)

        success = (
            stick_at_goal
            & ~is_left_grasping
            & ~is_right_grasping
            & is_right_static
        )

        return {
            "success": success,
            "is_left_grasping": is_left_grasping,
            "is_right_grasping": is_right_grasping,
            "stick_on_right_side": stick_on_right_side,
            "stick_lifted": stick_lifted,
            "stick_at_goal": stick_at_goal,
            "is_right_static": is_right_static,
            "goal_xy_dist": goal_xy_dist,
        }

    def _get_obs_extra(self, info: dict):
        obs = dict(
            left_arm_tcp=self.left_agent.tcp.pose.raw_pose,
            right_arm_tcp=self.right_agent.tcp.pose.raw_pose,
        )
        if "state" in self.obs_mode:
            obs.update(
                stick_pose=self.stick.pose.raw_pose,
                goal_pos=self._goal_xyz,
                left_tcp_to_stick=self.stick.pose.p - self.left_agent.tcp.pose.p,
                right_tcp_to_stick=self.stick.pose.p - self.right_agent.tcp.pose.p,
                stick_to_goal=self._goal_xyz - self.stick.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        stick_pos = self.stick.pose.p
        left_tcp = self.left_agent.tcp.pose.p
        right_tcp = self.right_agent.tcp.pose.p
        is_left_grasping = info["is_left_grasping"]
        is_right_grasping = info["is_right_grasping"]

        # Compute the stick's two ends in world coordinates from the actual
        # current orientation, so reward shaping is correct even after the
        # left arm rotates the stick during transit. The cylinder's local
        # long axis is +z, and we use that to pick the two ends. The end
        # whose world y is smaller is the "left end" the left arm should
        # grasp; the other is the "right end" the right arm should receive.
        stick_q = self.stick.pose.q
        local_axis = torch.tensor(
            [0.0, 0.0, self.stick_half_length], device=self.device
        ).expand(stick_pos.shape[0], -1)
        axis_world = quaternion_apply(stick_q, local_axis)
        end_a = stick_pos + axis_world
        end_b = stick_pos - axis_world
        a_is_left = (end_a[:, 1] <= end_b[:, 1]).unsqueeze(-1).float()
        left_end = a_is_left * end_a + (1 - a_is_left) * end_b
        right_end = (1 - a_is_left) * end_a + a_is_left * end_b

        # ---------------------------------------------------------------
        # Stage 1: Left arm reaches the stick and grasps it.
        # Target the actual -y end so the +y end protrudes for the receiver.
        left_grasp_target = 0.7 * left_end + 0.3 * stick_pos
        left_to_stick = torch.linalg.norm(left_tcp - left_grasp_target, axis=1)
        reach_left = 1 - torch.tanh(5 * left_to_stick)
        reward = reach_left + 2.0 * is_left_grasping.float()

        # ---------------------------------------------------------------
        # Stage 2: Lift the stick off the left platform and carry it across
        # the gap, well above the platform tops, toward the +y side.
        stage_2_active = is_left_grasping
        # Encourage rising above transit clearance.
        lift_reward = 1 - torch.tanh(
            5 * torch.clamp(self.transit_min_z - stick_pos[:, 2], min=0.0)
        )
        # Encourage moving stick toward +y (right side).
        cross_reward = 1 - torch.tanh(
            5 * torch.clamp(self.gap_half_width - stick_pos[:, 1], min=0.0)
        )
        stage_2_reward = lift_reward + cross_reward
        reward[stage_2_active] = 3 + stage_2_reward[stage_2_active]

        # ---------------------------------------------------------------
        # Stage 3: With the stick lifted and crossing into the right side,
        # the right arm approaches the +y end and forms a good pre-grasp.
        stage_3_active = (
            is_left_grasping & info["stick_lifted"] & info["stick_on_right_side"]
        )
        right_grasp_target = 0.7 * right_end + 0.3 * stick_pos
        right_to_stick = torch.linalg.norm(right_tcp - right_grasp_target, axis=1)
        right_reach = 1 - torch.tanh(5 * right_to_stick)

        right_tip1 = self.right_agent.finger1_link.pose.p
        right_tip2 = self.right_agent.finger2_link.pose.p
        tip_height = 1 - torch.tanh(
            5 * torch.abs(right_tip1[:, 2] - right_tip2[:, 2])
        )
        tip_width = 1 - torch.tanh(
            5
            * torch.abs(
                torch.linalg.norm(right_tip1 - right_tip2, axis=1) - 0.07
            )
        )
        right_approaching = (right_to_stick < 0.15).float()
        tip_reward = (tip_height + tip_width) / 2 * right_approaching
        stage_3_reward = right_reach + tip_reward + 2.0 * is_right_grasping.float()
        reward[stage_3_active] = 5 + stage_3_reward[stage_3_active]

        # ---------------------------------------------------------------
        # Stage 4: Right is grasping; left should release and retreat.
        stage_4_active = is_right_grasping
        gripper_width = (self.left_agent.robot.get_qlimits()[0, -1, 1] * 2).to(
            self.device
        )
        ungrasp_reward_left = (
            torch.sum(self.left_agent.robot.get_qpos()[:, -2:], axis=1)
            / gripper_width
        )
        ungrasp_reward_left[~is_left_grasping] = 1.0
        left_arm_retreat = 1 - torch.tanh(
            torch.linalg.norm(
                self.left_agent.robot.get_qpos()[:, :-2]
                - self.left_init_qpos[:, :-2],
                axis=1,
            )
        )
        # Keep stick lifted while left releases.
        keep_lifted = 1 - torch.tanh(
            5 * torch.clamp(self.transit_min_z - stick_pos[:, 2], min=0.0)
        )
        stage_4_reward = ungrasp_reward_left + left_arm_retreat + keep_lifted
        reward[stage_4_active] = 9 + stage_4_reward[stage_4_active]

        # ---------------------------------------------------------------
        # Stage 5: Right arm has sole possession (left not grasping, right is)
        # -> bring stick to goal on right platform.
        stage_5_active = is_right_grasping & ~is_left_grasping
        stick_to_goal = torch.linalg.norm(stick_pos - self._goal_xyz, axis=1)
        place_reward = 1 - torch.tanh(5 * stick_to_goal)
        stage_5_reward = 2 * place_reward + left_arm_retreat
        reward[stage_5_active] = 13 + stage_5_reward[stage_5_active]

        # ---------------------------------------------------------------
        # Stage 6: Stick at goal -> release and stay still.
        stage_6_active = info["stick_at_goal"] & ~is_left_grasping
        ungrasp_reward_right = (
            torch.sum(self.right_agent.robot.get_qpos()[:, -2:], axis=1)
            / gripper_width
        )
        ungrasp_reward_right[~is_right_grasping] = 1.0
        right_static_reward = 1 - torch.tanh(
            5
            * torch.linalg.norm(
                self.right_agent.robot.get_qvel()[..., :-2], axis=1
            )
        )
        stage_6_reward = ungrasp_reward_right + right_static_reward
        reward[stage_6_active] = 17 + stage_6_reward[stage_6_active]

        reward[info["success"]] = 20

        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 20
