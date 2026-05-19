"""
Load an SAC checkpoint, roll out one episode, and print the policy's
entropy + per-dim log_std at every step.

Useful for diagnosing whether SAC's auto-tune collapsed the entropy
(esp. on the gripper dimensions) and learning stalled.
"""
from collections import defaultdict
from dataclasses import dataclass
import os
import random
import time
from typing import Optional

from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import tyro

import mani_skill.envs  # registers envs


@dataclass
class Args:
    exp_name: Optional[str] = None
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    capture_video: bool = False
    checkpoint: Optional[str] = None

    env_id: str = "MultiRobotPassStick-v1"
    num_eval_envs: int = 1
    eval_partial_reset: bool = False
    num_eval_steps: int = 100
    eval_reconfiguration_freq: Optional[int] = 1
    control_mode: Optional[str] = "pd_ee_delta_pose"


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod(), 256),
            nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
        )
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
        h, l = env.single_action_space.high, env.single_action_space.low
        self.register_buffer("action_scale", torch.tensor((h - l) / 2.0, dtype=torch.float32))
        self.register_buffer("action_bias",  torch.tensor((h + l) / 2.0, dtype=torch.float32))

    def forward(self, x):
        x = self.backbone(x)
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def get_eval_action(self, x):
        x_feat = self.backbone(x)
        mean = self.fc_mean(x_feat)
        return torch.tanh(mean) * self.action_scale + self.action_bias

    def sample_with_log_pi(self, x):
        """Same as SAC's get_action: sample tanh-squashed Gaussian and return log π(a|s)."""
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob, mean, log_std


def _unwrap_base_env(env):
    base = env.unwrapped
    while hasattr(base, "_env"):
        base = base._env
    return base


if __name__ == "__main__":
    args = tyro.cli(Args)
    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    results_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "results")
    )
    run_dir = os.path.join(results_root, args.env_id, "sac", run_name)
    os.makedirs(run_dir, exist_ok=True)

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    env_kwargs = dict(obs_mode="state", render_mode="rgb_array", sim_backend="gpu")
    if args.control_mode is not None:
        env_kwargs["control_mode"] = args.control_mode

    eval_envs = gym.make(
        args.env_id, num_envs=args.num_eval_envs,
        reconfiguration_freq=args.eval_reconfiguration_freq,
        human_render_camera_configs=dict(shader_pack="default"),
        **env_kwargs,
    )
    if isinstance(eval_envs.action_space, gym.spaces.Dict):
        eval_envs = FlattenActionSpaceWrapper(eval_envs)
    if args.capture_video:
        eval_output_dir = f"{run_dir}/videos"
        if args.checkpoint is not None:
            eval_output_dir = f"{os.path.dirname(args.checkpoint)}/test_videos_entropy"
        eval_envs = RecordEpisode(
            eval_envs, output_dir=eval_output_dir,
            save_trajectory=False, save_video=True,
            trajectory_name="entropy", max_steps_per_video=args.num_eval_steps,
            video_fps=30, info_on_video=True,
        )

    eval_envs = ManiSkillVectorEnv(
        eval_envs, args.num_eval_envs,
        ignore_terminations=not args.eval_partial_reset, record_metrics=True,
    )

    actor = Actor(eval_envs).to(device)
    if args.checkpoint is not None:
        ckpt = torch.load(args.checkpoint, map_location=device)
        actor.load_state_dict(ckpt["actor"])
    actor.eval()

    eval_envs.single_observation_space.dtype = np.float32
    eval_obs, _ = eval_envs.reset(seed=args.seed)

    act_dim = int(np.prod(eval_envs.single_action_space.shape))
    print(f"action_dim = {act_dim}")
    # pd_ee_delta_pose: per arm 6 (xyz+rpy) + 1 gripper = 7 each, total 14
    # pd_joint_delta_pos: per arm 7 + 1 gripper = 8 each, total 16
    if act_dim == 14:
        gripper_indices = [6, 13]
    elif act_dim == 16:
        gripper_indices = [7, 15]
    else:
        gripper_indices = [act_dim // 2 - 1, act_dim - 1]   # fallback: assume last of each half
    print(f"gripper action indices: {gripper_indices}")

    # SAC's target_entropy convention
    target_entropy = -act_dim
    print(f"target_entropy (default SAC) = {target_entropy}")
    print()

    header = (
        f"{'t':>3} | {'H_sample':>9} {'H_gauss':>9} | "
        f"{'lstd_mean':>10} {'lstd_min':>9} {'lstd_max':>9} | "
        f"{'lstd_gL':>9} {'lstd_gR':>9} | "
        f"{'a_gL':>7} {'a_gR':>7}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for t in range(args.num_eval_steps):
        with torch.no_grad():
            # Deterministic action for stepping the env
            action_eval = actor.get_eval_action(eval_obs)
            # Sample + log_pi to estimate entropy at THIS state
            _, log_pi, mean, log_std = actor.sample_with_log_pi(eval_obs)

        H_sample = (-log_pi.mean()).item()                       # SAC-style entropy estimate
        # Pure Gaussian entropy (ignores tanh squash; useful for "policy width" intuition)
        H_gauss  = (log_std.sum(dim=1) + 0.5 * act_dim * (1.0 + np.log(2 * np.pi))).mean().item()

        log_std_np = log_std[0].cpu().numpy()
        action_np  = action_eval[0].cpu().numpy()

        print(
            f"{t:3d} | {H_sample:9.3f} {H_gauss:9.3f} | "
            f"{log_std_np.mean():10.3f} {log_std_np.min():9.3f} {log_std_np.max():9.3f} | "
            f"{log_std_np[gripper_indices[0]]:9.3f} {log_std_np[gripper_indices[1]]:9.3f} | "
            f"{action_np[gripper_indices[0]]:+7.3f} {action_np[gripper_indices[1]]:+7.3f}"
        )

        rows.append({
            "t": t,
            "H_sample": H_sample, "H_gauss": H_gauss,
            "lstd_mean": float(log_std_np.mean()),
            "lstd_min":  float(log_std_np.min()),
            "lstd_max":  float(log_std_np.max()),
            "lstd_gripL": float(log_std_np[gripper_indices[0]]),
            "lstd_gripR": float(log_std_np[gripper_indices[1]]),
            "action_gripL": float(action_np[gripper_indices[0]]),
            "action_gripR": float(action_np[gripper_indices[1]]),
        })

        eval_obs, _, _, _, _ = eval_envs.step(action_eval)

    # Aggregate summary
    H_sample_arr = np.array([r["H_sample"] for r in rows])
    lstd_gL = np.array([r["lstd_gripL"] for r in rows])
    lstd_gR = np.array([r["lstd_gripR"] for r in rows])
    lstd_arm = np.array([r["lstd_mean"] for r in rows])

    print("\n=== Summary ===")
    print(f"target_entropy (SAC default)   : {target_entropy}")
    print(f"mean H_sample   (over rollout) : {H_sample_arr.mean():.3f}")
    print(f"mean H_gauss    (over rollout) : {np.mean([r['H_gauss'] for r in rows]):.3f}")
    print(f"H_sample range                 : [{H_sample_arr.min():.3f}, {H_sample_arr.max():.3f}]")
    print()
    print(f"log_std mean (all dims, all t) : {lstd_arm.mean():+.3f}  (std≈{np.exp(lstd_arm.mean()):.3f})")
    print(f"log_std gripper LEFT  mean     : {lstd_gL.mean():+.3f}  (std≈{np.exp(lstd_gL.mean()):.3f})")
    print(f"log_std gripper RIGHT mean     : {lstd_gR.mean():+.3f}  (std≈{np.exp(lstd_gR.mean()):.3f})")
    print()
    if lstd_gL.mean() < -3 or lstd_gR.mean() < -3:
        print("⚠  gripper log_std < -3  →  gripper policy is essentially DETERMINISTIC.")
        print("   This is the smoking gun: SAC has stopped exploring gripper actions.")
    if H_sample_arr.mean() > -target_entropy:
        print("⚠  policy entropy > -target_entropy  →  SAC will keep pushing alpha → 0.")
        print("   (arm dims are over-jittering while gripper dims are collapsed.)")

    eval_envs.close()
