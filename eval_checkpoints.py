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
import tensorflow as tf
import dill

from contrastive import utils as contrastive_utils

BASEDIR = "/iris/u/khatch/contrastive_rl/results/contrastive_rl_goals3"

def eval(basedir, headdir):
    env_name_pieces = headdir.split("/")[0].split("-")
    env_name = "-".join(env_name_pieces)
    env, obs_dim = contrastive_utils.make_environment(env_name, start_index=0, end_index=-1, seed=0)

    print("\n\nenv._environment._environment._environment:", env._environment._environment._environment, "\n\n")

    checkpoint_files = glob(os.path.join(basedir, headdir, "seed_0", "checkpoints", "learner", "ckpt-*.index"))
    checkpoint_files = sorted(checkpoint_files, key=lambda x:int(x.split("/")[-1].split("-")[-1].split(".")[0]))

    for checkpoint_file in checkpoint_files:
        reader = tf.train.load_checkpoint(checkpoint_file)
        # reader = tf.train.load_checkpoint(os.path.join(basedir, headdir, "seed_0", "checkpoints", "learner"))
        params = reader.get_tensor('learner/.ATTRIBUTES/py_state')
        state = dill.loads(params)

        import pdb; pdb.set_trace()

def parse_arguments():
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Save Episode Images")
    parser.add_argument("--headdir", type=str,  help="")
    # parser.add_argument("--colors", action="store_true", default=False, help="")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    eval(BASEDIR, args.headdir)

"""
python3 -u eval_checkpoints.py \
--headdir fetch_push-goals-no-noise/learner/nonoise_collect_entropy
"""
