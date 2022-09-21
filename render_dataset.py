import os
import numpy as np
from tqdm import tqdm, trange
from glob import glob
import matplotlib.pyplot as plt
from mujoco_py import MjSimState
from tqdm import tqdm, trange
import cv2
import imageio
import io

from contrastive import utils as contrastive_utils

BASEDIR = "/iris/u/khatch/contrastive_rl/results/contrastive_rl_goals3"

def render_data(basedir, headdir):
    env, obs_dim = contrastive_utils.make_environment(headdir.split("/")[0] + "_image", start_index=0, end_index=-1, seed=0)

    print("\n\nenv._environment._environment._environment:", env._environment._environment._environment, "\n\n")
    datadir = os.path.join(basedir, headdir, "seed_0", "recorded_data")

    returns = []
    any_success = []
    end_success = []
    end_5_success = []

    episode_files = glob(os.path.join(datadir, "*.npz"))
    get_ep_no = lambda x:int(x.split("/")[-1].split(".")[0].split("-")[-1])
    episode_files = sorted(episode_files, key=get_ep_no)
    # j = 0
    for episode_file in tqdm(episode_files, total=len(episode_files), desc="Loading episode files"):
        # j += 1
        # if j > 1000:
        #     break
        env._sample_goal()
        env.reset()
        with open(episode_file, 'rb') as f:
            episode = np.load(f, allow_pickle=True)
            episode = {k: episode[k] for k in episode.keys()}

        assert len(episode["observation"]) == len(episode["step_type"]) == len(episode["action"])  == len(episode["discount"]) == len(episode["reward"])

        returns.append(episode["reward"].sum())
        any_success.append(episode["reward"].sum() > 0)
        end_success.append(episode["reward"][-1].sum() > 0)
        end_5_success.append(episode["reward"][-5:].sum() > 0)

    #     writer = []
        images = np.zeros((episode["observation"].shape[0], 64 * 64 * 3), dtype=np.uint8)
        for t in range(episode["observation"].shape[0]):


            sim_state = MjSimState(time=episode["sim_state"][t]["time"],
                               qpos=episode["sim_state"][t]["qpos"],
                               qvel=episode["sim_state"][t]["qvel"],
                               act=episode["sim_state"][t]["act"],
                               udd_state=episode["sim_state"][t]["udd_state"])
            env.sim.set_state(sim_state)
            img = env.observation(None)
            images[t] = img

    #         img_reshaped = np.reshape(img.copy(), (64, 64, 3))
    #         print("img.shape:", img.shape)
    #         print("img.dtype:", img.dtype)
    #         print("img_reshaped.shape:", img_reshaped.shape)
    #         img_reshaped = cv2.resize(img_reshaped, (512, 512))
    #         writer.append(img_reshaped)

        episode["image"] = images
        episode["goal_image"] = env._goal_img.copy().flatten()
        with io.BytesIO() as f1:
            np.savez_compressed(f1, **episode)
            f1.seek(0)
            with open(episode_file, 'wb') as f2:
                f2.write(f1.read())
    #     imageio.mimsave('output.gif', writer)
    #     break

    print(f"Number of episodes: {len(returns)}")
    print(f"[Return] mean: {np.mean(returns):.3f} max: {np.max(returns):.3f} min: {np.min(returns):.3f} std: {np.std(returns):.3f}")
    print(f"[Any Success] num: {np.sum(any_success)}/{len(any_success)} percent: {(np.sum(any_success) / len(any_success) * 100):.3f}%")
    print(f"[End Success] num: {np.sum(end_success)}/{len(end_success)} percent: {(np.sum(end_success) / len(end_success) * 100):.3f}%")
    print(f"[End 5 Success] num: {np.sum(end_5_success)}/{len(end_5_success)} percent: {(np.sum(end_5_success) / len(end_5_success) * 100):.3f}%")


def parse_arguments():
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Save Episode Images")
    parser.add_argument("--headdir", type=str,  help="")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    render_data(BASEDIR, args.headdir)

"""
python3 -u render_dataset.py \
--headdir fetch_reach-goals-no-noise/learner/nonoise_collect_alr=1e-5,clr=1e-5_minstd0.1_entropy

python3 -u render_dataset.py \
--headdir fetch_reach-goals-no-noise/learner/nonoise_collect_entropy
"""
