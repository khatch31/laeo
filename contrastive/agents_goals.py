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

"""Defines distributed contrastive RL agents, using JAX."""

import functools
from typing import Callable, Optional, Sequence

from acme import specs
from acme.jax import utils
from acme.utils import loggers
from contrastive import builder_goals
from contrastive import config as contrastive_config
from contrastive import distributed_layout_goals
from contrastive import networks
from contrastive import utils as contrastive_utils

import dm_env

import numpy as np


from contrastive.default_logger import make_default_logger

NetworkFactory = Callable[[specs.EnvironmentSpec],
                          networks.ContrastiveNetworks]

class DistributedContrastiveGoals(distributed_layout_goals.DistributedLayoutGoals):
  """Distributed program definition for contrastive RL."""

  def __init__(
      self,
      environment_factory,
      network_factory,
      config,
      seed,
      num_actors,
      max_number_of_steps = None,
      log_to_bigtable = False,
      log_every = 10.0,
      evaluator_factories = None,
      expert_goals=None,
      logdir=None,
      wandblogger=None,
      save_data=False,
      save_sim_state=False,
      data_save_dir="~/acme/data",
      data_load_dir=None,
      reward_checkpoint_state=None
  ):
    # Check that the environment-specific parts of the config have been set.
    assert config.max_episode_steps > 0
    assert config.obs_dim > 0

    self._obs_dim = config.obs_dim

    self._expert_goals = expert_goals
    self._logdir = logdir
    self._wandblogger = wandblogger
    self._data_load_dir = data_load_dir
    self._reward_checkpoint_state = reward_checkpoint_state

    logger_fn = functools.partial(make_default_logger,
                                  self._logdir,
                                  'learner', log_to_bigtable,
                                  time_delta=log_every, asynchronous=True,
                                  serialize_fn=utils.fetch_devicearray,
                                  steps_key='learner_steps',
                                  wandblogger=wandblogger)
    contrastive_builder = builder_goals.ContrastiveBuilderGoals(config,
                                                     logger_fn=logger_fn,
                                                     # save_data=save_data,
                                                     save_data=False,
                                                     data_save_dir=data_save_dir)
    if evaluator_factories is None:
      eval_policy_factory = (
          lambda n: networks.apply_policy_and_sample(n, True))
      eval_observers = [
          contrastive_utils.SuccessObserver(),
          contrastive_utils.LastNSuccessObserver(1),
          contrastive_utils.LastNSuccessObserver(5),
          contrastive_utils.LastNSuccessObserver(10),
          contrastive_utils.DistanceObserver(
              obs_dim=config.obs_dim,
              start_index=config.start_index,
              end_index=config.end_index),
          contrastive_utils.SavingObserver(
              data_save_dir,
              save=save_data,
              save_sim_state=save_sim_state)
      ]

      if config.log_video:
          eval_observers.append(contrastive_utils.VideoObserver(render_size=(512, 512), log_freq=config.video_log_freq, fps=20, video_format="mp4"))

      if "fixedxpos" in config.env_name:
          in_distribution = np.array([0.3, 0.2, 0.0, -0.1, -0.4])
          interpolation = np.array([0.1, -0.2, -0.3])
          extrapolation = np.array([0.5, 0.4, -0.5])
          eval_observers.append(contrastive_utils.PerGoalSuccessObserver(in_distribution=in_distribution, interpolation=interpolation, extrapolation=extrapolation))

      evaluator_factories = [
          distributed_layout_goals.default_evaluator_factory(
              environment_factory=environment_factory,
              network_factory=network_factory,
              policy_factory=eval_policy_factory,
              log_to_bigtable=log_to_bigtable,
              observers=eval_observers,
              logdir=self._logdir,
              wandblogger=self._wandblogger)
      ]
      if config.local:
        evaluator_factories = []
    actor_observers = [
        contrastive_utils.SuccessObserver(),
        contrastive_utils.LastNSuccessObserver(1),
        contrastive_utils.LastNSuccessObserver(5),
        contrastive_utils.LastNSuccessObserver(10),
        contrastive_utils.DistanceObserver(obs_dim=config.obs_dim,
                                           start_index=config.start_index,
                                           end_index=config.end_index)]

    super().__init__(
        seed=seed,
        environment_factory=environment_factory,
        network_factory=network_factory,
        builder=contrastive_builder,
        policy_network=networks.apply_policy_and_sample,
        evaluator_factories=evaluator_factories,
        num_actors=num_actors,
        max_number_of_steps=max_number_of_steps,
        prefetch_size=config.prefetch_size,
        log_to_bigtable=log_to_bigtable,
        actor_logger_fn=distributed_layout_goals.get_default_logger_fn(
            log_to_bigtable, log_every, self._logdir, self._wandblogger),
        observers=actor_observers,
        # checkpointing_config=distributed_layout_goals.CheckpointingConfig(),
        checkpointing_config=distributed_layout_goals.CheckpointingConfig(directory=self._logdir, max_to_keep=config.max_checkpoints_to_keep, add_uid=False),)
