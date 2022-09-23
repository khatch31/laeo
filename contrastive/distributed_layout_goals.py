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

"""Program definition for a distributed layout based on a builder."""

import dataclasses
import logging
from typing import Any, Callable, Optional, Sequence

from acme import core
from acme import environment_loop
from acme import specs
from acme.agents.jax import builders
from acme.jax import networks as networks_lib
from acme.jax import savers
from acme.jax import types
from acme.jax import utils
from acme.utils import counting
from acme.utils import loggers
from acme.utils import lp_utils
from acme.utils import observers as observers_lib
import dm_env
import jax
import launchpad as lp
import numpy as np
import reverb
import tqdm

from contrastive.default_logger import make_default_logger
from glob import glob
import os



ActorId = int
AgentNetwork = Any
PolicyNetwork = Any
NetworkFactory = Callable[[specs.EnvironmentSpec], AgentNetwork]
PolicyFactory = Callable[[AgentNetwork], PolicyNetwork]
Seed = int
EnvironmentFactory = Callable[[Seed], dm_env.Environment]
MakeActorFn = Callable[[types.PRNGKey, PolicyNetwork, core.VariableSource],
                       core.Actor]
LoggerLabel = str
LoggerStepsKey = str
LoggerFn = Callable[[LoggerLabel, LoggerStepsKey], loggers.Logger]
EvaluatorFactory = Callable[[
    types.PRNGKey,
    core.VariableSource,
    counting.Counter,
    MakeActorFn,
], core.Worker]


def get_default_logger_fn(
    log_to_bigtable = False,
    log_every = 10,
    logdir="~/acme",
    wandblogger=None): ###===### ###---###
  """Creates an actor logger."""

  def create_logger(actor_id):
    return make_default_logger(
        logdir, ###===### ###---###
        'actor',
        save_data=(log_to_bigtable and actor_id == 0),
        time_delta=log_every,
        steps_key='actor_steps',
        wandblogger=wandblogger)
  return create_logger


def default_evaluator_factory(
    environment_factory,
    network_factory,
    policy_factory,
    observers = (),
    log_to_bigtable = False,
    logdir="~/acme",
    wandblogger=None): ###===### ###---###
  """Returns a default evaluator process."""
  def evaluator(
      random_key,
      variable_source,
      counter,
      make_actor,
  ):
    """The evaluation process."""

    # Create environment and evaluator networks
    environment_key, actor_key = jax.random.split(random_key)
    # Environments normally require uint32 as a seed.
    environment = environment_factory(utils.sample_uint32(environment_key))
    networks = network_factory(specs.make_environment_spec(environment))

    actor = make_actor(actor_key, policy_factory(networks), variable_source)

    # Create logger and counter.
    counter = counting.Counter(counter, 'evaluator')
    logger = make_default_logger(logdir, 'evaluator', log_to_bigtable,
                                         steps_key='actor_steps',
                                         wandblogger=wandblogger)

    # Create the run loop and return it.
    return environment_loop.EnvironmentLoop(environment, actor, counter,
                                            logger, observers=observers)
  return evaluator


@dataclasses.dataclass
class CheckpointingConfig:
    def __init__(self, max_to_keep=1, directory="~/acme", add_uid=True):
        self.max_to_keep = max_to_keep
        self.directory = directory
        self.add_uid = add_uid ###===### ###---###

  # """Configuration options for learner checkpointer."""
  # # The maximum number of checkpoints to keep.
  # max_to_keep: int = 1
  # # Which directory to put the checkpoint in.
  # directory: str = '~/acme'
  # # If True adds a UID to the checkpoint path, see
  # # `paths.get_unique_id()` for how this UID is generated.
  # add_uid: bool = True


class DistributedLayoutGoals:
  """Program definition for a distributed agent based on a builder."""

  def __init__(
      self,
      seed,
      environment_factory,
      network_factory,
      builder,
      policy_network,
      num_actors,
      environment_spec = None,
      actor_logger_fn = None,
      evaluator_factories = (),
      device_prefetch = True,
      prefetch_size = 1,
      log_to_bigtable = False,
      max_number_of_steps = None,
      observers = (),
      multithreading_colocate_learner_and_reverb = False,
      checkpointing_config = None):

    if prefetch_size < 0:
      raise ValueError(f'Prefetch size={prefetch_size} should be non negative')

    actor_logger_fn = actor_logger_fn or get_default_logger_fn(log_to_bigtable)

    self._seed = seed
    self._builder = builder
    self._environment_factory = environment_factory
    self._network_factory = network_factory
    self._policy_network = policy_network
    self._environment_spec = environment_spec
    self._num_actors = num_actors
    self._device_prefetch = device_prefetch
    self._log_to_bigtable = log_to_bigtable
    self._prefetch_size = prefetch_size
    self._max_number_of_steps = max_number_of_steps
    self._actor_logger_fn = actor_logger_fn
    self._evaluator_factories = evaluator_factories
    self._observers = observers
    self._multithreading_colocate_learner_and_reverb = (
        multithreading_colocate_learner_and_reverb)
    self._checkpointing_config = checkpointing_config

  def replay(self):
    """The replay storage."""
    dummy_seed = 1
    environment_spec = (
        self._environment_spec or
        specs.make_environment_spec(self._environment_factory(dummy_seed)))
    return self._builder.make_replay_tables(environment_spec)

  def counter(self):
    kwargs = {}
    if self._checkpointing_config:
      kwargs = vars(self._checkpointing_config)
    return savers.CheckpointingRunner(
        counting.Counter(),
        key='counter',
        subdirectory='counter',
        time_delta_minutes=5,
        **kwargs)

  def learner(
      self,
      random_key,
      replay,
      counter,
      expert_goals, ###===### ###---###
  ):
    """The Learning part of the agent."""

    dummy_seed = 1
    environment_spec = (
        self._environment_spec or
        specs.make_environment_spec(self._environment_factory(dummy_seed)))

    # Creates the networks to optimize (online) and target networks.
    networks = self._network_factory(environment_spec)

    if self._builder._config.env_name.startswith('offline_ant'):  # pytype: disable=attribute-error, pylint: disable=protected-access
      adder = self._builder.make_adder(replay, force_no_save=True)
      env = self._environment_factory(0)
      dataset = env.get_dataset()  # pytype: disable=attribute-error
      for t in tqdm.trange(dataset['observations'].shape[0]):
        discount = 1.0
        if t == 0 or dataset['timeouts'][t - 1]:
          step_type = dm_env.StepType.FIRST
        elif dataset['timeouts'][t]:
          step_type = dm_env.StepType.LAST
          discount = 0.0
        else:
          step_type = dm_env.StepType.MID

        ts = dm_env.TimeStep(
            step_type=step_type,
            reward=dataset['rewards'][t],
            discount=discount,
            observation=np.concatenate([dataset['observations'][t],
                                        dataset['infos/goal'][t]]),
        )
        if t == 0 or dataset['timeouts'][t - 1]:
          adder.add_first(ts)  # pytype: disable=attribute-error
        else:
          adder.add(action=dataset['actions'][t-1], next_timestep=ts)  # pytype: disable=attribute-error

        if self._builder._config.local and t > 10_000:  # pytype: disable=attribute-error, pylint: disable=protected-access
          break


    if self._builder._config.use_td:
        assert self._reward_checkpoint_state is not None
        dataset_r_preds = []
        sigmoid_dataset_r_preds = []

    use_image_obs = self._builder._config.use_image_obs
    if self._builder._config.env_name.startswith('offline_fetch') or self._builder._config.env_name.startswith('offline_push'):
        assert self._data_load_dir is not None
        adder = self._builder.make_adder(replay, force_no_save=True)

        if expert_goals is None:
            expert_goals_list = []

        episode_files = glob(os.path.join(self._data_load_dir, "*.npz"))
        get_ep_no = lambda x:int(x.split("/")[-1].split(".")[0].split("-")[-1])
        episode_files = sorted(episode_files, key=get_ep_no)
        # episode_files = sorted(episode_files, key=get_ep_no, reverse=True)
        # j = 0
        for episode_file in tqdm.tqdm(episode_files, total=len(episode_files), desc="Loading episode files"):
            # j += 1
            # if j > 500:
            #     break
            with open(episode_file, 'rb') as f:
                episode = np.load(f, allow_pickle=True)
                episode = {k: episode[k] for k in episode.keys()}

            assert len(episode["observation"]) == len(episode["step_type"]) == len(episode["action"])  == len(episode["discount"]) == len(episode["reward"])
            if use_image_obs:
                assert len(episode["observation"]) == len(episode["image"])

            if expert_goals is None and episode["reward"].sum() > 0:
                success_idxs = np.nonzero(episode["reward"])[0]
                # success_observations = episode["observation"][success_idxs]
                for idx in success_idxs:
                    if use_image_obs:
                        assert episode["image"][idx].shape[0] == self._obs_dim # Should be the same regardless of slicing, goal image stored seperately in data
                        expert_goals_list.append(episode["image"][idx][:self._obs_dim])#.astype(np.float32))
                    else:
                        expert_goals_list.append(episode["observation"][idx][:self._obs_dim])

            for t in range(episode["observation"].shape[0]):
                if use_image_obs:
                    obs = np.concatenate((episode['image'][t], episode['goal_image']), axis=0)#.astype(np.float32)
                else:
                    obs = episode['observation'][t]

                if self._builder._config.use_td:
                    r_pred = networks.r_network.apply(self._reward_checkpoint_state.r_params, episode["observation"][t, :self._obs_dim])
                    dataset_r_preds.append(r_pred.item())
                    sigmoid_dataset_r_preds.append(jax.nn.sigmoid(r_pred).item())

                ts = dm_env.TimeStep(
                    step_type=episode["step_type"][t],
                    reward=episode['reward'][t] if not self._builder._config.use_td else jax.nn.sigmoid(r_pred).item(),
                    discount=episode["discount"][t],
                    observation=obs,
                )
                if t == 0:
                    assert episode["step_type"][t] == dm_env.StepType.FIRST
                    adder.add_first(ts)  # pytype: disable=attribute-error
                else:
                    assert episode["step_type"][t] == dm_env.StepType.LAST if t == episode["observation"].shape[0] -1 else dm_env.StepType.MID
                    adder.add(action=episode['action'][t], next_timestep=ts)  # pytype: disable=attribute-error

        if expert_goals is None:
            N_EXAMPLES = 200
            idxs = np.arange(len(expert_goals_list))
            np.random.shuffle(idxs)
            idxs = idxs[:N_EXAMPLES]
            expert_goals = [expert_goals_list[i] for i in idxs]
            expert_goals = np.stack(expert_goals)

        if self._builder._config.use_td:
            expert_r_preds = [networks.r_network.apply(self._reward_checkpoint_state.r_params, expert_goals[i]).item() for i in range(expert_goals.shape[0])]
            exp_dataset_r_preds = [np.exp(r_pred) for r_pred in dataset_r_preds]
            exp_expert_r_preds = [np.exp(r_pred) for r_pred in expert_r_preds]
            sigmoid_expert_r_preds = [jax.nn.sigmoid(r_pred) for r_pred in expert_r_preds]

            print(f"\n[dataset_r_preds] mean: {np.mean(dataset_r_preds):.3f}, std: {np.std(dataset_r_preds):.3f}, max: {np.max(dataset_r_preds):.3f}, min: {np.min(dataset_r_preds):.3f},")
            print(f"[sigmoid_dataset_r_preds] mean: {np.mean(sigmoid_dataset_r_preds):.3f}, std: {np.std(sigmoid_dataset_r_preds):.3f}, max: {np.max(sigmoid_dataset_r_preds):.3f}, min: {np.min(sigmoid_dataset_r_preds):.3f},")
            print(f"[exp_dataset_r_preds] mean: {np.mean(exp_dataset_r_preds):.3f}, std: {np.std(exp_dataset_r_preds):.3f}, max: {np.max(exp_dataset_r_preds):.3f}, min: {np.min(exp_dataset_r_preds):.3f},")

            print(f"[\nexpert_r_preds] mean: {np.mean(expert_r_preds):.3f}, std: {np.std(expert_r_preds):.3f}, max: {np.max(expert_r_preds):.3f}, min: {np.min(expert_r_preds):.3f},")
            print(f"[sigmoid_expert_r_preds] mean: {np.mean(sigmoid_expert_r_preds):.3f}, std: {np.std(sigmoid_expert_r_preds):.3f}, max: {np.max(sigmoid_expert_r_preds):.3f}, min: {np.min(sigmoid_expert_r_preds):.3f},")
            print(f"[exp_expert_r_preds] mean: {np.mean(exp_expert_r_preds):.3f}, std: {np.std(exp_expert_r_preds):.3f}, max: {np.max(exp_expert_r_preds):.3f}, min: {np.min(exp_expert_r_preds):.3f},")

    iterator = self._builder.make_dataset_iterator(replay)

    # dummy_seed = 1
    # environment_spec = (
    #     self._environment_spec or
    #     specs.make_environment_spec(self._environment_factory(dummy_seed)))
    #
    # # Creates the networks to optimize (online) and target networks.
    # networks = self._network_factory(environment_spec)

    if self._prefetch_size > 1:
      # When working with single GPU we should prefetch to device for
      # efficiency. If running on TPU this isn't necessary as the computation
      # and input placement can be done automatically. For multi-gpu currently
      # the best solution is to pre-fetch to host although this may change in
      # the future.
      device = jax.devices()[0] if self._device_prefetch else None
      iterator = utils.prefetch(
          iterator, buffer_size=self._prefetch_size, device=device)
    else:
      logging.info('Not prefetching the iterator.')

    counter = counting.Counter(counter, 'learner')

    learner = self._builder.make_learner(random_key, networks, iterator, replay,
                                         counter, expert_goals) ###===### ###---###
    kwargs = {}
    if self._checkpointing_config:
      kwargs = vars(self._checkpointing_config)
    # Return the learning agent.
    return savers.CheckpointingRunner(
        learner,
        key='learner',
        subdirectory='learner',
        time_delta_minutes=5,
        **kwargs)

  def actor(self, random_key, replay,
            variable_source, counter,
            actor_id):
    """The actor process."""
    adder = self._builder.make_adder(replay)

    environment_key, actor_key = jax.random.split(random_key)
    # Create environment and policy core.

    # Environments normally require uint32 as a seed.
    environment = self._environment_factory(
        utils.sample_uint32(environment_key))

    networks = self._network_factory(specs.make_environment_spec(environment))
    policy_network = self._policy_network(networks)
    actor = self._builder.make_actor(actor_key, policy_network, adder,
                                     variable_source)

    # Create logger and counter.
    counter = counting.Counter(counter, 'actor')
    # Only actor #0 will write to bigtable in order not to spam it too much.
    logger = self._actor_logger_fn(actor_id)
    # Create the loop to connect environment and agent.
    return environment_loop.EnvironmentLoop(environment, actor, counter,
                                            logger, observers=self._observers)

  def coordinator(self, counter, max_actor_steps):
    if self._builder._config.env_name.startswith('offline'):  # pytype: disable=attribute-error, pylint: disable=protected-access
      steps_key = 'learner_steps'
    else:
      steps_key = 'actor_steps'
    return lp_utils.StepsLimiter(counter, max_actor_steps, steps_key=steps_key)

  def build(self, name='agent', program = None):
    """Build the distributed agent topology."""
    if not program:
      program = lp.Program(name=name)

    key = jax.random.PRNGKey(self._seed)

    def r_checpointer():
        import os
        from reverb.platform.checkpointers_lib import DefaultCheckpointer
        # return DefaultCheckpointer("/iris/u/khatch/contrastive_rl/results/trash_results/fetch_reach/learner/default/seed_0/checkpoints/replay_buffer")
        return DefaultCheckpointer(os.path.join(self._logdir, "checkpoints", "replay_buffer"))

    if self._builder._config.env_name.startswith('offline'):
        replay_node = lp.ReverbNode(self.replay)
    else:
        replay_node = lp.ReverbNode(self.replay, checkpoint_time_delta_minutes=5, checkpoint_ctor=r_checpointer)

    with program.group('replay'):
      if self._multithreading_colocate_learner_and_reverb:
        replay = replay_node.create_handle()
      else:
        replay = program.add_node(replay_node)

    with program.group('counter'):
      counter = program.add_node(lp.CourierNode(self.counter))
      if self._max_number_of_steps is not None:
        _ = program.add_node(
            lp.CourierNode(self.coordinator, counter,
                           self._max_number_of_steps))

    learner_key, key = jax.random.split(key)
    learner_node = lp.CourierNode(self.learner, learner_key, replay, counter, self._expert_goals) ###===### ###---###
    with program.group('learner'):
      if self._multithreading_colocate_learner_and_reverb:
        learner = learner_node.create_handle()
        program.add_node(
            lp.MultiThreadingColocation([learner_node, replay_node]))
      else:
        learner = program.add_node(learner_node)

    def make_actor(random_key,
                   policy_network,
                   variable_source):
      return self._builder.make_actor(
          random_key, policy_network, variable_source=variable_source)

    with program.group('evaluator'):
      for evaluator in self._evaluator_factories:
        evaluator_key, key = jax.random.split(key)
        program.add_node(
            lp.CourierNode(evaluator, evaluator_key, learner, counter,
                           make_actor))

    with program.group('actor'):
      for actor_id in range(self._num_actors):
        actor_key, key = jax.random.split(key)
        program.add_node(
            lp.CourierNode(self.actor, actor_key, replay, learner, counter,
                           actor_id))

    return program

"""
if t == 0:
  step_type = dm_env.StepType.FIRST
elif t == episode["observation"].shape[0] - 1:
  step_type = dm_env.StepType.LAST
  discount = 0.0
else:
  step_type = dm_env.StepType.MID

ts = dm_env.TimeStep(
    step_type=step_type,
    reward=episode['reward'][t],
    discount=episode["discount"][t],
    observation=episode['observation'][t],
)
if t == 0:
  adder.add_first(ts)  # pytype: disable=attribute-error
else:
  adder.add(action=episode['action'][t - 1], next_timestep=ts)  # pytype: disable=attribute-erro

"""
