"""Quick preview of TwoRobotPassStickTwoTables-v1: resets the env and saves
a render-camera image to ./preview_two_tables.png so you can eyeball the
table/gap/stick layout without launching the GUI.

Run:
    python -m mani_skill.envs.tasks.tabletop.preview_two_tables
"""
import gymnasium as gym
import imageio.v3 as iio
import numpy as np

import mani_skill.envs  # noqa: F401  (registers TwoRobotPassStickTwoTables-v1)


def main():
    env = gym.make(
        "TwoRobotPassStickTwoTables-v1",
        num_envs=1,
        obs_mode="state",
        render_mode="rgb_array",
        sim_backend="cpu",
    )
    env.reset(seed=0)
    img = env.unwrapped.render().cpu().numpy()[0]
    out = "preview_two_tables.png"
    iio.imwrite(out, img.astype(np.uint8))
    print(f"saved {out}  shape={img.shape}")
    env.close()


if __name__ == "__main__":
    main()
