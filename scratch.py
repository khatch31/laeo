def run_scratch():
    # env_test()
    # checkpoint_load()
    dataset_count_test()

def dataset_count_test():
    import tensorflow as tf
    FILENAMES = "/iris/u/khatch/contrastive_rl/results/contrastive_rl_goals/fetch_reach-goals-no-noise/learner/nonoise_2/seed_0/checkpoints/replay_buffer/2022-09-01T12:12:53.751428098-07:00"


    tf.compat.v1.enable_eager_execution

    x = sum(1 for _ in tf.data.TFRecordDataset(FILENAMES + "/tables.tfrecord"))
    import pdb; pdb.set_trace()

def env_test():
    import os
    import numpy as np
    import gym


    from fetch_envs import FetchReachEnv

    env = FetchReachEnv()
    import pdb; pdb.set_trace()


def checkpoint_load():
    import tensorflow as tf
    import dill
    import contrastive
    from contrastive import utils as contrastive_utils
    import jax
    import numpy as np
    from base64 import b64encode
    import tempfile
    import jax.numpy as jnp

    filename = '/iris/u/khatch/contrastive_rl/results/contrastive_rl_goals/fetch_reach/learner/default/seed_0/checkpoints/learner'
    reader = tf.train.load_checkpoint(filename)
    params = reader.get_tensor('learner/.ATTRIBUTES/py_state')
    state = dill.loads(params)

    import pdb; pdb.set_trace()

    state.q_params
    state.target_q_params
    state.q_optimizer_state




if __name__ == "__main__":
    run_scratch()


"""
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:=/iris/u/khatch/anaconda3/envs/contrastive_rl/lib/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/afs/cs.stanford.edu/u/khatch/.mujoco/mujoco200/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-000

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/afs/cs.stanford.edu/u/khatch/.mujoco/mujoco210/bin

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/iris/u/khatch/anaconda3/envs/contrastive_rl/lib/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/iris/u/khatch/.mujoco/mujoco200/bin
"""
