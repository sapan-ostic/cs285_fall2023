import os
import gym
import matplotlib.pyplot as plt
import imageio
import numpy as np

os.environ["MUJOCO_GL"] = "egl"

MJ_ENV_NAMES = ["Ant-v4", "Walker2d-v4", "HalfCheetah-v4", "Hopper-v4"]


def is_valid_frame(frame):
    return (
        isinstance(frame, np.ndarray)
        and frame.ndim == 3
        and frame.shape[2] in [1, 3, 4]  # valid channel counts
    )


def run_random_policy(params, num_episodes=1, max_steps=200):
    env = gym.make(params['env_name'], render_mode="rgb_array")

    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        step = 0
        frames = []

        while not done and step < max_steps:
            frame = env.render()
            if isinstance(frame, list):
                frame = frame[0]  # Grab the first camera view

            print(f"Frame at step {step}: type={type(frame)}, shape={getattr(frame, 'shape', None)}")

            if is_valid_frame(frame):
                frames.append(frame)

            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            step += 1
            print(f"Episode {episode + 1}, Step {step}, Reward: {reward}, Done: {done}")

        if frames:
            video_filename = f"rollout_{params['env_name']}_ep{episode + 1}.mp4"
            imageio.mimsave(video_filename, frames, fps=30)
            print(f"Saved video: {video_filename}")
        else:
            print("No valid frames captured — check rendering mode or backend.")

    env.close()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '-env', type=str, help=f'choices: {", ".join(MJ_ENV_NAMES)}', required=True)
    args = parser.parse_args()
    params = vars(args)
    run_random_policy(params)


if __name__ == "__main__":
    main()
