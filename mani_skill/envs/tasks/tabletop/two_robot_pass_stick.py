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


@register_env("TwoRobotPassStick-v1", max_episode_steps=100)
class TwoRobotPassStick(BaseEnv):
    """
    **Task Description:**
    Two robot arms must pass a stick between them across a gap between two
    separate raised tables. The stick lies on the left table with its long
    axis along world y (perpendicular to the gap). Because the gap is wider
    than the stick is long, the only way to move it from the left table to
    the right table is to lift it through the air — pushing it over the edge
    will let it fall into the gap. The left arm grasps the near (-y) end of
    the stick, lifts it across to a handoff above the gap, the right arm
    regrasps the protruding (+y) end, the left arm releases, and the right
    arm places the stick at a goal position on the right table.

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

    stick_half_width = 0.015
    stick_half_length = 0.13

    table_top_z = 0.0
    table_height = 0.9196429  # matches TableSceneBuilder default table height
    table_half_size_x = 0.75
    table_half_size_y = 0.75

    # Tables centered at y = ±0.90 with half_size_y = 0.75 give a 0.30 m gap
    # (inner edges at y = ±0.15). Arm bases sit at y = ±0.90 (see _load_agent).
    table_center_y_abs = 0.90

    @property
    def gap_half_width(self):
        return self.table_center_y_abs - self.table_half_size_y

    # Success thresholds.
    goal_radius = 0.06 

    # Left arm parks the stick for the right arm to regrasp.
    # Centered over the gap (y=0) so each arm reaches its OWN end of the
    # stick (ends at y=±0.13). With bases at y=±0.90 and z=0.25, that's
    # ~0.77 m horizontal / ~0.81 m 3D from base to end — about 5 cm of
    # margin under the 0.855 m max reach. The stick CENTER (y=0) sits in
    # the unreachable corridor for both arms at this height; only the
    # protruding end is reachable.
    handoff_xyz = (0.0, 0.0, 0.25)
    handoff_radius = 0.08  # stick is "at handoff" if center within this distance

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
        pose = sapien_utils.look_at(eye=[3.0, 0.0, 2.0], target=[0.0, 0.0, 0.1])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(
            options,
            [
                sapien.Pose(p=[0, -self.table_center_y_abs, 0]),
                sapien.Pose(p=[0, self.table_center_y_abs, 0]),
            ],
        )

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        # The default table will be moved out of sight in _initialize_episode
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
            self.table_scene.table.set_pose(self._default_table_hidden_pose)
            self.left_init_qpos = self.left_agent.robot.get_qpos()

            stick_xyz = torch.zeros((b, 3))
            stick_xyz[:, 0] = torch.rand((b,)) * 0.10 - 0.05  # x in [-0.05, 0.05]
            stick_xyz[:, 1] = -0.30 + torch.rand((b,)) * 0.02  # center y in [-0.30, -0.28]
            stick_xyz[:, 2] = self.table_top_z + self.stick_half_width

            self.stick.set_pose(Pose.create_from_pq(p=stick_xyz))

            goal_xyz = torch.zeros((b, 3))
            goal_xyz[:, 0] = torch.rand((b,)) * 0.10 - 0.05  # x in [-0.05, +0.05]
            goal_xyz[:, 1] = 1.40 + torch.rand((b,)) * 0.10  # y in [+1.40, +1.50]
            goal_xyz[:, 2] = self.table_top_z + 1e-3  # just above tabletop
            # Rotate so the disc lies flat
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

        handoff_target = torch.tensor(
            self.handoff_xyz, device=self.device, dtype=stick_pos.dtype
        )
        stick_at_handoff = (
            torch.linalg.norm(stick_pos - handoff_target, axis=1)
            <= self.handoff_radius
        )

        stick_at_goal = (
            torch.linalg.norm(stick_pos[:, :2] - goal_pos[:, :2], axis=1)
            <= self.goal_radius
        )

        success = stick_at_goal & ~is_left_grasping & ~is_right_grasping

        return {
            "success": success,
            "is_left_grasping": is_left_grasping,
            "is_right_grasping": is_right_grasping,
            "stick_at_handoff": stick_at_handoff,
            "stick_at_goal": stick_at_goal,
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

        gripper_width = (self.left_agent.robot.get_qlimits()[0, -1, 1] * 2).to(
            self.device
        )



        # Stage 1: Left arm reaches the stick with a pre-grasp tip-shape reward.
        # Target the -y end of the stick. 
        # Stage 1 max ≈ 1 + 1 + 2 = 4
        left_grasp_target = stick_pos.clone()
        left_grasp_target[:, 1] -= self.stick_half_length
        left_to_target = torch.linalg.norm(left_tcp - left_grasp_target, axis=1)
        reach_left = 1 - torch.tanh(5 * left_to_target)

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

        # Stage 2: With left grasping, bring the stick to the fixed handoff
        # target.
        # Stage 2 max ≈ 4 + 3 = 7
                
        stage_2_active = is_left_grasping
        handoff_target = torch.tensor(
            self.handoff_xyz, device=self.device, dtype=stick_pos.dtype
        )
        stick_to_handoff_dist = torch.linalg.norm(
            stick_pos - handoff_target, axis=1
        )
        handoff_reach = 1 - torch.tanh(5 * stick_to_handoff_dist)
        stage_2_reward = 3 * handoff_reach
        reward[stage_2_active] = 4 + stage_2_reward[stage_2_active]

        # ---------------------------------------------------------------
        # Stage 3: Stick is at the handoff. Aggressive shaping for the right
        # arm regrasp:
        #   - left_static: hold the stick steady (no rotation drift)
        #   - finger_proximity: each fingertip close to the stick body
        #   - 5x grasp bonus: closing the jaws pays off strongly
        # Stage 3 max ≈ 7 + 5 + 1 + 1 = 14

        # Penalize left-arm motion. 
        left_qvel = self.left_agent.robot.get_qvel()[:, :-2]
        left_static = 1 - torch.tanh(5 * torch.linalg.norm(left_qvel, axis=1))
        
        right_finger1 = self.right_agent.finger1_link.pose.p
        right_finger2 = self.right_agent.finger2_link.pose.p
        finger1_close = 1 - torch.tanh(
            10 * torch.linalg.norm(right_finger1 - stick_pos, axis=1)
        )
        finger2_close = 1 - torch.tanh(
            10 * torch.linalg.norm(right_finger2 - stick_pos, axis=1)
        )
        right_finger_proximity = (finger1_close + finger2_close) / 2

        stage_3_active = is_left_grasping & info["stick_at_handoff"]
        stage_3_reward = (
            3.0 * is_right_grasping.float()
            + left_static
            + right_finger_proximity
        )
        reward[stage_3_active] = 7 + stage_3_reward[stage_3_active]


        # ---------------------------------------------------------------
        # Stage 4: Right is grasping -> left releases and retreats home.
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
        stage_4_reward = ungrasp_reward_left + left_arm_retreat
        reward[stage_4_active] = 14 + stage_4_reward[stage_4_active]
        # Stage 4 max ≈ 14 + 2 = 16

        # ---------------------------------------------------------------
        # Stage 5: Right has sole possession -> bring the stick to the goal
        # on the right table.
        stage_5_active = is_right_grasping & ~is_left_grasping
        stick_to_goal = torch.linalg.norm(stick_pos - self.goal_region.pose.p, axis=1)
        # Goal is ~1.5 m from the handoff: slope 1.0 keeps the curve responsive
        # across that journey instead of saturating at the start.
        place_reward = 1 - torch.tanh(1.0 * stick_to_goal)
        stage_5_reward = 2 * place_reward + left_arm_retreat
        reward[stage_5_active] = 16 + stage_5_reward[stage_5_active]
        # Stage 5 max ≈ 16 + 3 = 19

        # ---------------------------------------------------------------
        # Stage 6: Stick is on the goal target -> right releases.
        stage_6_active = info["stick_at_goal"] & ~is_left_grasping
        ungrasp_reward_right = (
            torch.sum(self.right_agent.robot.get_qpos()[:, -2:], axis=1)
            / gripper_width
        )
        ungrasp_reward_right[~is_right_grasping] = 1.0
        stage_6_reward = ungrasp_reward_right
        reward[stage_6_active] = 19 + stage_6_reward[stage_6_active]
        # Stage 6 max ≈ 19 + 1 = 20

        # ---------------------------------------------------------------
        # Always-on right-arm shaping: pre-position the right TCP at the
        # exact pose needed to grasp the +y end at handoff (2 cm inside the
        # end, 3 cm above the stick top — top-down grasp pose). Fixed point
        # in world coords, reachable, so the right arm gets a clean gradient
        # toward the right place from step 1.
        right_grasp_target = torch.tensor(
            [
                0.0,
                self.handoff_xyz[1] + self.stick_half_length - 0.02,
                self.handoff_xyz[2] + 0.03,
            ],
            device=self.device,
            dtype=stick_pos.dtype,
        ).expand(stick_pos.shape[0], -1)
        right_to_target = torch.linalg.norm(right_tcp - right_grasp_target, axis=1)
        right_reach = 1 - torch.tanh(2.0 * right_to_target)
        # Once grasping, full credit so this term doesn't drag the arm back
        # to the handoff while it's trying to place the stick at the goal.
        right_reach = right_reach.clone()
        right_reach[is_right_grasping] = 1.0

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
        right_tip_reward = (right_tip_height + right_tip_width) / 2
        # Once grasping, give full credit so closing the gripper on the stick
        # doesn't fight the grasp bonus.
        right_tip_reward = right_tip_reward.clone()
        right_tip_reward[is_right_grasping] = 1.0

        reward = reward + right_reach + right_tip_reward
        # Total max ≈ 20 + 2 = 22

        reward[info["success"]] = 25

        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 25
