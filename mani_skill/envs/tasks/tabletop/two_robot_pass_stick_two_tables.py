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


@register_env("TwoRobotPassStickTwoTables-v1", max_episode_steps=100)
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
    - The stick's xy position is within the goal disc (red/white target on
      the right table; success if xy distance <= goal_radius).
    - Neither arm is grasping the stick.
    """

    SUPPORTED_ROBOTS = [("panda_wristcam", "panda_wristcam")]
    agent: MultiAgent[Tuple[Panda, Panda]]

    # Stick geometry: a box-shaped bar (won't roll like a cylinder).
    # 3cm x 3cm cross-section, 26cm long along its local y-axis. Total length
    # 0.26 < gap (0.40), so the stick cannot bridge the gap by being pushed.
    stick_half_width = 0.015
    stick_half_length = 0.13

    # Two physically separate tables, one per robot, with a gap along y wider
    # than the stick is long. Each table's top is at z = 0. The gap is empty 
    # space. If the stick is dropped over it, it falls to the floor.
    table_top_z = 0.0
    table_height = 0.9196429  # matches TableSceneBuilder default table height
    table_half_size_x = 0.50
    table_half_size_y = 0.50
    # Each table is centered on its robot base (which TableSceneBuilder
    # anchors at y=±0.75) so the base sits at the center of its table box.
    # Inner edges sit at y=±0.25 -> gap = 0.50 > stick (0.26).
    table_center_y_abs = 0.75

    @property
    def gap_half_width(self):
        # Inner edge of each table along y; the empty gap spans (-x, +x).
        return self.table_center_y_abs - self.table_half_size_y

    # Success thresholds.
    goal_radius = 0.06  # red/white target disc radius on the right table

    # Mid-air clearance threshold (stick must rise above this while crossing the gap).
    transit_min_z = 0.06

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
        # Pulled back along +x and raised to fit both tables (y spans ~2.4m)
        # plus both robots into the frame.
        pose = sapien_utils.look_at(eye=[1.8, 0.0, 1.4], target=[0.0, 0.0, 0.05])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(
            options, [sapien.Pose(p=[0, -1, 0]), sapien.Pose(p=[0, 1, 0])]
        )

    def _load_scene(self, options: dict):
        # We use TableSceneBuilder for the ground plane and the two-Panda
        # qpos / base-pose initialization, but we don't want its single big
        # table — we want two separate tables with a gap. So we build it,
        # then move the default table out of sight, and add our own tables.
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        # The default table will be moved out of sight in _initialize_episode
        # (set_pose can't run safely here — GPU rigid-body buffers aren't
        # allocated yet during _load_scene).
        self._default_table_hidden_pose = sapien.Pose(p=[0, 0, -100.0])

        self._left_table_y = -self.table_center_y_abs
        self._right_table_y = self.table_center_y_abs
        # Center z so the top sits at table_top_z = 0 and bottom reaches floor.
        table_center_z = self.table_top_z - self.table_height / 2

        table_half_sizes = [
            self.table_half_size_x,
            self.table_half_size_y,
            self.table_height / 2,
        ]
        self.left_table = actors.build_box(
            self.scene,
            half_sizes=table_half_sizes,
            color=[0.7, 0.55, 0.35, 1.0],
            name="left_table",
            body_type="kinematic",
            initial_pose=sapien.Pose(
                p=[0.0, self._left_table_y, table_center_z]
            ),
        )
        self.right_table = actors.build_box(
            self.scene,
            half_sizes=table_half_sizes,
            color=[0.7, 0.55, 0.35, 1.0],
            name="right_table",
            body_type="kinematic",
            initial_pose=sapien.Pose(
                p=[0.0, self._right_table_y, table_center_z]
            ),
        )

        # Stick: a rectangular bar. Long axis is the box's local y, so the
        # stick naturally lies along world y with identity orientation —
        # no transient rotation needed. Flat faces prevent rolling.
        self.stick = actors.build_box(
            self.scene,
            half_sizes=[
                self.stick_half_width,
                self.stick_half_length,
                self.stick_half_width,
            ],
            color=[0.6, 0.3, 0.1, 1.0],
            name="stick",
            initial_pose=sapien.Pose(p=[0, self._left_table_y, 0.3]),
        )

        # Goal marker: a red/white target disc lying flat on the right
        # table (visual only — no collision with the stick).
        self.goal_region = actors.build_red_white_target(
            self.scene,
            radius=self.goal_radius,
            thickness=1e-5,
            name="goal_region",
            add_collision=False,
            body_type="kinematic",
            initial_pose=sapien.Pose(),
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            # Hide the default single table — we use our own two tables.
            # (table_scene.initialize re-poses the default table each reset,
            # so we override that pose here every reset.)
            self.table_scene.table.set_pose(self._default_table_hidden_pose)
            self.left_init_qpos = self.left_agent.robot.get_qpos()

            # Place the stick lying along y on the left table, forward of
            # the left arm toward the gap (left arm base at y=-0.75; the
            # gap edge is at y=-0.25). Tight x range like PickCube.
            stick_xyz = torch.zeros((b, 3))
            stick_xyz[:, 0] = torch.rand((b,)) * 0.10 - 0.05   # x in [-0.05, +0.05]
            # y range chosen so the *low-y end* (stick.y - 0.13) clears the
            # left base front edge (~y=-0.65) and the *high-y end* stays on
            # the table (inner edge y=-0.25).
            stick_xyz[:, 1] = -0.52 + torch.rand((b,)) * 0.12  # y in [-0.52, -0.40]
            stick_xyz[:, 2] = self.table_top_z + self.stick_half_width + 0.002

            # Box's long axis is already its local y -> identity orientation.
            stick_q = (
                torch.tensor(
                    [1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=self.device
                )
                .unsqueeze(0)
                .expand(b, -1)
            )
            self.stick.set_pose(Pose.create_from_pq(p=stick_xyz, q=stick_q))

            # Goal: on the right table, *behind* the right arm in +y, well
            # clear of the base footprint (right arm base at y=+0.75 with
            # ~0.07 footprint; goal disc radius 0.06, so goal center y must
            # be >= 0.75 + 0.07 + 0.06 ≈ 0.88 to fully clear).
            goal_xyz = torch.zeros((b, 3))
            goal_xyz[:, 0] = torch.rand((b,)) * 0.10 - 0.05  # x in [-0.05, +0.05]
            goal_xyz[:, 1] = 1.05 + torch.rand((b,)) * 0.10  # y in [+1.00, +1.10]
            goal_xyz[:, 2] = self.table_top_z + 1e-3  # just above tabletop
            # Rotate so the disc lies flat (StackCube convention).
            self.goal_region.set_pose(
                Pose.create_from_pq(
                    p=goal_xyz,
                    q=euler2quat(0, np.pi / 2, 0),
                )
            )

    @property
    def left_agent(self) -> Panda:
        return self.agent.agents[0]

    @property
    def right_agent(self) -> Panda:
        return self.agent.agents[1]

    def evaluate(self):
        stick_pos = self.stick.pose.p
        goal_pos = self.goal_region.pose.p

        is_left_grasping = self.left_agent.is_grasping(self.stick)
        is_right_grasping = self.right_agent.is_grasping(self.stick)

        stick_on_right_side = stick_pos[:, 1] >= self.gap_half_width
        stick_lifted = stick_pos[:, 2] >= self.transit_min_z

        # Stick is "at" the goal target if its xy is within the disc radius.
        goal_xy_dist = torch.linalg.norm(
            stick_pos[:, :2] - goal_pos[:, :2], axis=1
        )
        stick_at_goal = goal_xy_dist <= self.goal_radius

        success = stick_at_goal & ~is_left_grasping & ~is_right_grasping

        return {
            "success": success,
            "is_left_grasping": is_left_grasping,
            "is_right_grasping": is_right_grasping,
            "stick_on_right_side": stick_on_right_side,
            "stick_lifted": stick_lifted,
            "stick_at_goal": stick_at_goal,
            "goal_xy_dist": goal_xy_dist,
        }

    def _get_obs_extra(self, info: dict):
        obs = dict(
            left_arm_tcp=self.left_agent.tcp.pose.raw_pose,
            right_arm_tcp=self.right_agent.tcp.pose.raw_pose,
        )
        if "state" in self.obs_mode:
            goal_pos = self.goal_region.pose.p
            obs.update(
                stick_pose=self.stick.pose.raw_pose,
                goal_pos=goal_pos,
                left_tcp_to_stick=self.stick.pose.p - self.left_agent.tcp.pose.p,
                right_tcp_to_stick=self.stick.pose.p - self.right_agent.tcp.pose.p,
                stick_to_goal=goal_pos - self.stick.pose.p,
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
        # left arm rotates the stick during transit. The box's local long
        # axis is +y, and we use that to pick the two ends. The end whose
        # world y is smaller is the "left end" the left arm should grasp;
        # the other is the "right end" the right arm should receive.
        stick_q = self.stick.pose.q
        local_axis = torch.tensor(
            [0.0, self.stick_half_length, 0.0], device=self.device
        ).expand(stick_pos.shape[0], -1)
        axis_world = quaternion_apply(stick_q, local_axis)
        end_a = stick_pos + axis_world
        end_b = stick_pos - axis_world
        a_is_left = (end_a[:, 1] <= end_b[:, 1]).unsqueeze(-1).float()
        left_end = a_is_left * end_a + (1 - a_is_left) * end_b
        right_end = (1 - a_is_left) * end_a + a_is_left * end_b

        gripper_width = (self.left_agent.robot.get_qlimits()[0, -1, 1] * 2).to(
            self.device
        )

        # ---------------------------------------------------------------
        # Stage 1: Left arm reaches the stick's low-y end (the one closest
        # to itself), biased to that end but with a center component so the
        # gradient stays smooth far away. Plus an always-on pre-grasp
        # tip-shape reward to give the policy an orientation gradient.
        # Target the stick *center* (PickCube/StackCube style): a fat,
        # smooth attractor that's far easier to learn than end-targeting.
        # Where exactly the agent grasps along the stick is left to emerge;
        # the +y/-y rotational symmetry is broken in stage 2 instead.
        left_to_stick = torch.linalg.norm(left_tcp - stick_pos, axis=1)
        reach_left = 1 - torch.tanh(5 * left_to_stick)

        left_tip1 = self.left_agent.finger1_link.pose.p
        left_tip2 = self.left_agent.finger2_link.pose.p
        left_tip_height = 1 - torch.tanh(
            5 * torch.abs(left_tip1[:, 2] - left_tip2[:, 2])
        )
        left_tip_width = 1 - torch.tanh(
            5
            * torch.abs(
                torch.linalg.norm(left_tip1 - left_tip2, axis=1) - 0.07
            )
        )
        left_tip_reward = (left_tip_height + left_tip_width) / 2

        reward = reach_left + left_tip_reward + 2.0 * is_left_grasping.float()
        # Stage 1 max ≈ 1 + 1 + 2 = 4

        # ---------------------------------------------------------------
        # Stage 2: With left grasping, lift the stick over the gap and push
        # its y-coordinate past the right-table inner edge. PickCube-style
        # 1D positional threshold for `cross` — only the left arm can move
        # the stick, so this is a pure "left-does-its-job" signal with no
        # right-arm hacking and no dependence on stick orientation.
        stage_2_active = is_left_grasping
        lift_reward = 1 - torch.tanh(
            5 * torch.clamp(self.transit_min_z - stick_pos[:, 2], min=0.0)
        )
        # Slope 1.0 (not 5.0): the stick must travel ~0.70 m to satisfy
        # this reward. With slope=5, tanh saturates immediately and the
        # policy gets no gradient toward +y — it can flip the wrist
        # backward at zero reward cost. Slope=1 keeps the gradient
        # informative across the full approach distance.
        cross_reward = 1 - torch.tanh(
            1.0 * torch.clamp(self.gap_half_width - stick_pos[:, 1], min=0.0)
        )
        # Weight lift > cross so the agent learns to lift the stick before
        # moving it sideways across the gap (otherwise it slides into the gap).
        stage_2_reward = 2 * lift_reward + cross_reward
        reward[stage_2_active] = 4 + stage_2_reward[stage_2_active]
        # Stage 2 max ≈ 4 + 3 = 7

        # ---------------------------------------------------------------
        # Stage 3: Stick is lifted and on the right side -> right arm
        # approaches the stick (target the center, like PickCube — a fat
        # attractor, not the end) with a good pre-grasp, then grasps it.
        stage_3_active = (
            is_left_grasping & info["stick_lifted"] & info["stick_on_right_side"]
        )
        # Slope 1.0 (not 5.0): when stage 3 first activates the right TCP is
        # ~1 m from the stick (right base at y=+0.75, stick just past y=-0.25).
        # tanh(5 * 1.0) saturates near 1 -> reward ≈ 0 with no gradient toward
        # the stick, so the right arm has no incentive to approach. Slope 1
        # keeps the gradient informative across the full approach distance.
        right_to_stick = torch.linalg.norm(right_tcp - stick_pos, axis=1)
        right_reach = 1 - torch.tanh(1.0 * right_to_stick)

        right_tip1 = self.right_agent.finger1_link.pose.p
        right_tip2 = self.right_agent.finger2_link.pose.p
        right_tip_height = 1 - torch.tanh(
            5 * torch.abs(right_tip1[:, 2] - right_tip2[:, 2])
        )
        right_tip_width = 1 - torch.tanh(
            5
            * torch.abs(
                torch.linalg.norm(right_tip1 - right_tip2, axis=1) - 0.07
            )
        )
        # Always-on tip preshape (PickCube pattern), no <0.15 m gate.
        right_tip_reward = (right_tip_height + right_tip_width) / 2
        # Hold-still reward: without this, the left arm (which holds the
        # stick) earns `right_reach` by dragging the stick toward the right
        # TCP at the same time the right arm approaches the stick — both
        # move and the system oscillates. Penalizing stick + left-arm
        # velocity makes the left arm a stationary handoff target so the
        # right arm has to do the approaching.
        stick_lin_vel = torch.linalg.norm(self.stick.linear_velocity, axis=1)
        stick_ang_vel = torch.linalg.norm(self.stick.angular_velocity, axis=1)
        left_qvel = torch.linalg.norm(
            self.left_agent.robot.get_qvel()[:, :-2], axis=1
        )
        hold_still = (
            (1 - torch.tanh(5 * stick_lin_vel))
            + (1 - torch.tanh(stick_ang_vel))
            + (1 - torch.tanh(left_qvel))
        ) / 3
        stage_3_reward = (
            right_reach
            + right_tip_reward
            + hold_still
            + 2.0 * is_right_grasping.float()
        )
        reward[stage_3_active] = 7 + stage_3_reward[stage_3_active]
        # Stage 3 max ≈ 7 + 1 + 1 + 1 + 2 = 12

        # ---------------------------------------------------------------
        # Stage 4: Right is grasping -> left releases, retreats home, and
        # stick stays clear of the gap.
        stage_4_active = is_right_grasping
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
        keep_lifted = 1 - torch.tanh(
            5 * torch.clamp(self.transit_min_z - stick_pos[:, 2], min=0.0)
        )
        stage_4_reward = ungrasp_reward_left + left_arm_retreat + keep_lifted
        reward[stage_4_active] = 12 + stage_4_reward[stage_4_active]
        # Stage 4 max ≈ 12 + 3 = 15

        # ---------------------------------------------------------------
        # Stage 5: Right has sole possession -> bring the stick to the goal
        # on the right table.
        stage_5_active = is_right_grasping & ~is_left_grasping
        stick_to_goal = torch.linalg.norm(stick_pos - self.goal_region.pose.p, axis=1)
        place_reward = 1 - torch.tanh(5 * stick_to_goal)
        stage_5_reward = 2 * place_reward + left_arm_retreat
        reward[stage_5_active] = 15 + stage_5_reward[stage_5_active]
        # Stage 5 max ≈ 15 + 2 + 1 = 18

        # ---------------------------------------------------------------
        # Stage 6: Stick is on the goal target -> right releases.
        stage_6_active = info["stick_at_goal"] & ~is_left_grasping
        ungrasp_reward_right = (
            torch.sum(self.right_agent.robot.get_qpos()[:, -2:], axis=1)
            / gripper_width
        )
        ungrasp_reward_right[~is_right_grasping] = 1.0
        stage_6_reward = ungrasp_reward_right
        reward[stage_6_active] = 18 + stage_6_reward[stage_6_active]
        # Stage 6 max ≈ 18 + 1 = 19

        reward[info["success"]] = 20

        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 20
