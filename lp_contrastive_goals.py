# coding=utf-8
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# python3
r"""Example running contrastive RL in JAX.

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:=/iris/u/khatch/anaconda3/envs/contrastive_rl/lib/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/afs/cs.stanford.edu/u/khatch/.mujoco/mujoco200/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-000
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/afs/cs.stanford.edu/u/khatch/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia


export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:=/iris/u/khatch/anaconda3/envs/crl2/lib/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/afs/cs.stanford.edu/u/khatch/.mujoco/mujoco200/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-000
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/afs/cs.stanford.edu/u/khatch/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

python3 -u lp_contrastive_goals.py \
--lp_launch_type=local_mt \
--env_name=fetch_reach-goals \
--logdir=/iris/u/khatch/contrastive_rl/trash_results

Run using multi-processing (required for image-based experiments):
  python lp_contrastive.py --lp_launch_type=local_mp

Run using multi-threading
  python lp_contrastive.py --lp_launch_type=local_mt
"""
import functools
from typing import Any, Dict

from absl import app
from absl import flags
import contrastive
from contrastive import utils as contrastive_utils
import launchpad as lp

import os
import shutil
import tensorflow as tf
import dill

from contrastive.wandb_logger import WANDBLogger


FLAGS = flags.FLAGS
flags.DEFINE_bool('debug', False, 'Runs training for just a few steps.')
flags.DEFINE_string('env_name', None, 'Env_name.')
flags.DEFINE_string('logdir', "~/acme", 'Env_name.')
flags.DEFINE_string('description', "default", 'description.')
flags.DEFINE_string('project', "contrastive_rl_goals", 'description.')
flags.DEFINE_string('replay_buffer_load_dir', None, 'description.')
flags.DEFINE_float('entropy_coefficient', None, 'description.')
flags.DEFINE_integer('num_actors', 4, 'description.')
flags.DEFINE_bool('invert_actor_loss', False, 'description.')
flags.DEFINE_bool('exp_q_action', False, 'description.')

flags.DEFINE_integer('max_number_of_steps', 1_000_000, 'description.')
flags.DEFINE_integer('batch_size', 256, 'description.')
flags.DEFINE_float('actor_learning_rate', 3e-4, 'description.')
flags.DEFINE_float('learning_rate', 3e-4, 'description.')
flags.DEFINE_integer('num_sgd_steps_per_step', 64, 'description.')
flags.DEFINE_integer('repr_dim', 64, 'description.')
flags.DEFINE_integer('max_replay_size', 1000000, 'description.')
flags.DEFINE_multi_integer('hidden_layer_sizes', [256, 256], 'description.')
flags.DEFINE_float('actor_min_std', 1e-6, 'description.')

flags.DEFINE_bool('save_data', False, 'description.')
flags.DEFINE_string('data_load_dir', None, 'description.')
flags.DEFINE_integer('max_checkpoints_to_keep', 1, 'description.')

flags.DEFINE_float('bc_coef', 0, 'description.')
flags.DEFINE_bool('twin_q', True, 'description.')

flags.DEFINE_bool('save_sim_state', False, 'description.')
flags.DEFINE_bool('use_gcbc', False, 'description.')

flags.DEFINE_bool('use_td', False, 'description.')
flags.DEFINE_string('reward_checkpoint_path', None, 'description.')


@functools.lru_cache()
def get_env(env_name, start_index, end_index):
  return contrastive_utils.make_environment(env_name, start_index, end_index,
                                            seed=0)


def get_program(params):
  """Constructs the program."""

  # if FLAGS.save_data:
  #     assert params["num_actors"] == 1

  env_name = params['env_name']
  seed = params.pop('seed')

  if params.get('use_image_obs', False) and not params.get('local', False):
  #   print('WARNING: overwriting parameters for image-based tasks.')
    # params['num_sgd_steps_per_step'] = 16
    params['prefetch_size'] = 16
    # params['num_actors'] = 10

  if env_name.startswith('offline'):
    # No actors needed for the offline RL experiments. Evaluation is
    # handled separately.
    params['num_actors'] = 0
    assert not FLAGS.save_data

  config = contrastive.ContrastiveConfigGoals(**params)

  print("config.num_sgd_steps_per_step:", config.num_sgd_steps_per_step)

  env_factory = lambda seed: contrastive_utils.make_environment(  # pylint: disable=g-long-lambda
      env_name, config.start_index, config.end_index, seed)

  env_factory_no_extra = lambda seed: env_factory(seed)[0]  # Remove obs_dim.
  environment, obs_dim = get_env(env_name, config.start_index,
                                 config.end_index)
  assert (environment.action_spec().minimum == -1).all()
  assert (environment.action_spec().maximum == 1).all()
  config.obs_dim = obs_dim
  config.max_episode_steps = getattr(environment, '_step_limit') + 1
  if env_name == 'offline_ant_umaze_diverse':
    # This environment terminates after 700 steps, but demos have 1000 steps.
    config.max_episode_steps = 1000
  network_factory = functools.partial(
      contrastive.make_networks, obs_dim=obs_dim, repr_dim=config.repr_dim,
      repr_norm=config.repr_norm, twin_q=config.twin_q,
      use_image_obs=config.use_image_obs,
      hidden_layer_sizes=config.hidden_layer_sizes,
      actor_min_std=config.actor_min_std,
      use_td=config.use_td)

  expert_goals = environment.get_expert_goals()
  print("\nexpert_goals:\n", expert_goals)
  print(f"\nenvironment._environment._environment._environment: {environment._environment._environment._environment}")
  print(f"environment._environment._environment._environment._add_goal_noise: {environment._environment._environment._environment._add_goal_noise}\n\n")
  if "image" in env_name and "push" in env_name:
      print(f"environment._environment._environment._environment._rand_y: {environment._environment._environment._environment._rand_y}\n\n")

  algo = "learner_goals"
  if FLAGS.use_gcbc:
      algo = "bc"
  elif FLAGS.use_td:
      algo = "td"

  logdir = os.path.join(FLAGS.logdir, FLAGS.project, params["env_name"], algo, FLAGS.description, f"seed_{seed}")

  group_name="_".join([params["env_name"], algo, FLAGS.description])
  name=f"seed_{seed}"
  wandblogger = WANDBLogger(os.path.join(logdir, "wandb_logs"),
                            params,
                            group_name,
                            name,
                            FLAGS.project)

  if FLAGS.replay_buffer_load_dir is not None:
      os.makedirs(os.path.join(logdir, "checkpoints"), exist_ok=True)
      shutil.copytree(FLAGS.replay_buffer_load_dir, os.path.join(logdir, "checkpoints", "replay_buffer"))

  if config.use_td:
      assert FLAGS.reward_checkpoint_path is not None

  if FLAGS.reward_checkpoint_path is not None:
      assert config.use_td
      reader = tf.train.load_checkpoint(FLAGS.reward_checkpoint_path)
      params = reader.get_tensor('learner/.ATTRIBUTES/py_state')
      reward_checkpoint_state = dill.loads(params)
  else:
      assert not config.use_td
      reward_checkpoint_state = None

  agent = contrastive.DistributedContrastiveGoals(
      seed=seed,
      environment_factory=env_factory_no_extra,
      network_factory=network_factory,
      config=config,
      num_actors=config.num_actors,
      log_to_bigtable=True,
      max_number_of_steps=config.max_number_of_steps,
      expert_goals=expert_goals,
      logdir=logdir,
      wandblogger=wandblogger,
      save_data=FLAGS.save_data,
      save_sim_state=FLAGS.save_sim_state,
      data_save_dir=os.path.join(logdir, "recorded_data"),
      data_load_dir=FLAGS.data_load_dir,
      reward_checkpoint_state=reward_checkpoint_state)
  print("Done with agent init.")

  return agent.build()


def main(_):
  # Create experiment description.

  # 1. Select an environment.
  # Supported environments:
  #   Metaworld: sawyer_{push,drawer,bin,window}
  #   OpenAI Gym Fetch: fetch_{reach,push}
  #   D4RL AntMaze: ant_{umaze,,medium,large},
  #   2D nav: point_{Small,Cross,FourRooms,U,Spiral11x11,Maze11x11}
  # Image observation environments:
  #   Metaworld: sawyer_image_{push,drawer,bin,window}
  #   OpenAI Gym Fetch: fetch_{reach,push}_image
  #   2D nav: point_image_{Small,Cross,FourRooms,U,Spiral11x11,Maze11x11}
  # Offline environments:
  #   antmaze: offline_ant_{umaze,umaze_diverse,
  #                             medium_play,medium_diverse,
  #                             large_play,large_diverse}
  # env_name = 'sawyer_window' ###===###
  # env_name = 'fixed-goal-point_Cross' ###---###
  # env_name = "fetch_reach"

  if FLAGS.env_name:
      env_name = FLAGS.env_name

  params = {
      'seed': 0,
      'use_random_actor': True,
      'entropy_coefficient': None if 'image' in env_name else 0.0,
      'env_name': env_name,
      'max_number_of_steps': 1_000_000,
      'use_image_obs': 'image' in env_name,
  }

  params["max_checkpoints_to_keep"] = FLAGS.max_checkpoints_to_keep
  params["entropy_coefficient"] = FLAGS.entropy_coefficient
  params["num_actors"] = FLAGS.num_actors
  params["invert_actor_loss"] = FLAGS.invert_actor_loss
  params["exp_q_action"] = FLAGS.exp_q_action

  params["max_number_of_steps"] = FLAGS.max_number_of_steps
  params["batch_size"] = FLAGS.batch_size
  params["actor_learning_rate"] = FLAGS.actor_learning_rate
  params["learning_rate"] = FLAGS.learning_rate
  params["num_sgd_steps_per_step"] = FLAGS.num_sgd_steps_per_step
  params["repr_dim"] = FLAGS.repr_dim
  params["max_replay_size"] = FLAGS.max_replay_size
  params["hidden_layer_sizes"] = FLAGS.hidden_layer_sizes
  params["actor_min_std"] = FLAGS.actor_min_std

  params["bc_coef"] = FLAGS.bc_coef
  params["twin_q"] = FLAGS.twin_q

  params["use_gcbc"] = FLAGS.use_gcbc

  params["use_td"] = FLAGS.use_td
  # params["reward_checkpoint_path"] = FLAGS.reward_checkpoint_path

  if 'ant_' in env_name:
    params['end_index'] = 2

  # 2. Select an algorithm. The currently-supported algorithms are:
  # contrastive_nce, contrastive_cpc, c_learning, nce+c_learning, gcbc.
  # Many other algorithms can be implemented by passing other parameters
  # or adding a few lines of code.
  alg = 'contrastive_nce'
  if alg == 'contrastive_nce':
    pass  # Just use the default hyperparameters
  elif alg == 'contrastive_cpc':
    params['use_cpc'] = True
  elif alg == 'c_learning':
    params['use_td'] = True
    params['twin_q'] = True
  elif alg == 'nce+c_learning':
    params['use_td'] = True
    params['twin_q'] = True
    params['add_mc_to_td'] = True
  elif alg == 'gcbc':
    params['use_gcbc'] = True
  else:
    raise NotImplementedError('Unknown method: %s' % alg)

  if env_name.startswith('offline_fetch'):
    assert FLAGS.data_load_dir is not None

    params.update({
        # Effectively remove the rate-limiter by using very large values.
        'samples_per_insert': 1_000_000,
        'samples_per_insert_tolerance_rate': 100_000_000.0,
        'random_goals': 0.0,
    })

  # For the offline RL experiments, modify some hyperparameters.
  if env_name.startswith('offline_ant'):
    params.update({
        # Effectively remove the rate-limiter by using very large values.
        'samples_per_insert': 1_000_000,
        'samples_per_insert_tolerance_rate': 100_000_000.0,
        # For the actor update, only use future states as goals.
        'random_goals': 0.0,
        'bc_coef': 0.05,  # Add a behavioral cloning term to the actor.
        'twin_q': True,  # Learn two critics, and take the minimum.
        'batch_size': 1024,  # Increase the batch size 256 --> 1024.
        'repr_dim': 16,  # Decrease the representation size 64 --> 16.
        # Increase the policy network size (256, 256) --> (1024, 1024)
        'hidden_layer_sizes': (1024, 1024),
    })

  # 3. Select compute parameters. The default parameters are already tuned, so
  # use this mainly for debugging.
  if FLAGS.debug:
    params.update({
        'min_replay_size': 10_000,
        'local': True,
        'num_sgd_steps_per_step': 1,
        'prefetch_size': 1,
        'num_actors': 1,
        'batch_size': 32,
        'max_number_of_steps': 10_000,
        'samples_per_insert_tolerance_rate': 1.0,
        'hidden_layer_sizes': (32, 32),
    })

  program = get_program(params)
  # Set terminal='tmux' if you want different components in different windows.

  lp.launch(program, terminal='current_terminal')
  # local_resources = dict(actor=lp.PythonProcess(env=dict(XLA_PYTHON_CLIENT_MEM_FRACTION='0.1')))
  # lp.launch(program, terminal='current_terminal', local_resources=local_resources)

if __name__ == '__main__':
  app.run(main)
