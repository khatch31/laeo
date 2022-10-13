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
import pickle
from collections import defaultdict
import dm_env

from contrastive import utils as contrastive_utils


def format_data(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)

    returns = []
    any_success = []
    end_success = []
    end_5_success = []

    # dataset_metrics = defaultdict(list)
    data_files = list(glob(os.path.join(src_dir, "**", "ep-*pkl"), recursive=True))
    get_ep_no = lambda x:int(x.split("/")[-1].split(".")[0].split("-")[-1])
    data_files = sorted(data_files, key=get_ep_no)
    _, episode_length, obs_dim, act_dim, img_dim = _calc_n_episodes_and_length(data_files)

    idx = 0
    for episode_file in tqdm(data_files, desc="Loading CEBORL data episode files", total=len(data_files)):
        # dict_keys(['step_type', 'reward', 'discount', 'observation', 'action', 'sim_state', 'image', 'goal_image'])
        ep = load_pickled_object(episode_file)

        action = np.zeros((len(ep), act_dim), dtype=np.float32)
        step_type = np.zeros((len(ep)), dtype=np.int64)
        reward = np.zeros((len(ep),), dtype=np.float64)
        discount = np.zeros((len(ep)), dtype=np.float64)
        observation = np.zeros((len(ep), obs_dim * 2), dtype=np.float32)

        success = np.zeros((len(ep),), dtype=np.float64)
        image = np.zeros((len(ep), img_dim), dtype=np.uint8)
        sim_state = np.array([None for _ in range(len(ep))])

        # episode['action'][t].dtype: float32
        # ts.step_type.dtype: int64
        # ts.reward.dtype: float64
        # ts.discount.dtype: float64
        # ts.observation.dtype: float32

        assert len(ep) == episode_length, f"len(ep){len(ep)} != episode_length: {episode_length}"

        for t, save_traj in enumerate(ep):
            image[t] = np.squeeze(save_traj.image.numpy()).flatten()
            obs = np.squeeze(save_traj.traj.observation.numpy())
            obs = np.concatenate((obs, np.zeros_like(obs)), axis=0)
            observation[t] = obs
            action[t] = np.squeeze(save_traj.traj.action.numpy())
            discount[t] = save_traj.traj.discount.numpy().item()
            reward[t] = save_traj.traj.reward.numpy().item()
            success[t] = save_traj.info["success"].item()
            step_type[t] = convert_to_dm_step_type(save_traj.traj.step_type.numpy().item())
            sim_state[t] = {key:np.squeeze(val) for key, val in save_traj.info.items()}

        # dataset_metrics["success"].append(success.sum())
        # dataset_metrics["any_success"].append(success.sum() >= 1)
        # dataset_metrics["end_success"].append(success[-1] >= 1)
        # dataset_metrics["end_5_success"].append(success[-5:].sum() >= 1)
        # dataset_metrics["end_10_success"].append(success[-10:].sum() >= 1)
        # dataset_metrics["end_20_success"].append(success[-20:].sum() >= 1)
        # dataset_metrics["return"].append(rewards.sum())
        # dataset_metrics["length"].append(len(ep))

        episode = dict(reward=reward,
                       success=success,
                       discount=discount,
                       observation=observation,
                       action=action,
                       image=image,
                       goal_image=np.zeros_like(image[0]),
                       sim_state=sim_state,
                       step_type=step_type)

        returns.append(episode["reward"].sum())
        any_success.append(episode["reward"].sum() > 0)
        end_success.append(episode["reward"][-1].sum() > 0)
        end_5_success.append(episode["reward"][-5:].sum() > 0)

        dst_episode_file = os.path.join(dst_dir, f"ep-{get_ep_no(episode_file)}.npz")

        with io.BytesIO() as f1:
            np.savez_compressed(f1, **episode)
            f1.seek(0)
            with open(dst_episode_file, 'wb') as f2:
                f2.write(f1.read())

    print(f"Number of episodes: {len(returns)}")
    print(f"[Return] mean: {np.mean(returns):.3f} max: {np.max(returns):.3f} min: {np.min(returns):.3f} std: {np.std(returns):.3f}")
    print(f"[Any Success] num: {np.sum(any_success)}/{len(any_success)} percent: {(np.sum(any_success) / len(any_success) * 100):.3f}%")
    print(f"[End Success] num: {np.sum(end_success)}/{len(end_success)} percent: {(np.sum(end_success) / len(end_success) * 100):.3f}%")
    print(f"[End 5 Success] num: {np.sum(end_5_success)}/{len(end_5_success)} percent: {(np.sum(end_5_success) / len(end_5_success) * 100):.3f}%")

def convert_to_dm_step_type(step_type):
    # FIRST = 0, MID = 1, LAST = 2
    if step_type == 0:
        return dm_env.StepType.FIRST

    if step_type == 1:
        return dm_env.StepType.MID

    if step_type == 2:
        return dm_env.StepType.LAST

    raise ValueError(f"Unsupported step type: \"{step_type}\".")

def _calc_n_episodes_and_length(data_files):
    ep = load_pickled_object(data_files[0])
    obs_dim = ep[0].traj.observation.shape[-1]
    act_dim = ep[0].traj.action.shape[-1]
    img_dim = np.prod(ep[0].image.shape)
    assert img_dim == 12288
    return len(data_files), len(ep), obs_dim, act_dim, img_dim

def load_pickled_object(filepath, gzipped=False):
    if gzipped:
        with gzip.open(filepath, "rb") as f:
            obj = pickle.load(f)
    else:
        with open(filepath, "rb") as f:
            obj = pickle.load(f)

    return obj

def parse_arguments():
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Save Episode Images")
    parser.add_argument("--src_dir", type=str,  help="")
    parser.add_argument("--dst_dir", type=str,  help="")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    format_data(args.src_dir, args.dst_dir)



"""
python3 -u reformat_ceborl_data.py \
--src_dir /iris/u/khatch/crce/data/multimodal_metaworld/n_step_sac_data/original_view/dial_turn/medium_replay \
--dst_dir /iris/u/khatch/contrastive_rl/data/ceborl/dial_turn/medium_replay

python3 -u reformat_ceborl_data.py \
--src_dir /iris/u/khatch/crce/data/multimodal_metaworld/n_step_sac_data/original_view/door_open/medium_replay \
--dst_dir /iris/u/khatch/contrastive_rl/data/ceborl/door_open/medium_replay

python3 -u reformat_ceborl_data.py \
--src_dir /iris/u/khatch/crce/data/multimodal_metaworld/n_step_sac_data/original_view/drawer_open/medium_replay \
--dst_dir /iris/u/khatch/contrastive_rl/data/ceborl/drawer_open/medium_replay

python3 -u reformat_ceborl_data.py \
--src_dir /iris/u/khatch/crce/data/multimodal_metaworld/n_step_sac_data/original_view/lever_pull/medium_replay \
--dst_dir /iris/u/khatch/contrastive_rl/data/ceborl/lever_pull/medium_replay

python3 -u reformat_ceborl_data.py \
--src_dir /iris/u/khatch/crce/data/multimodal_metaworld/n_step_sac_data/original_view/plate_slide/medium_replay \
--dst_dir /iris/u/khatch/contrastive_rl/data/ceborl/plate_slide/medium_replay
"""
