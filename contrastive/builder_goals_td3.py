"""TD3 Builder."""
from typing import Iterator, List, Optional

from acme import adders
from acme import core
from acme import specs
from acme.adders import reverb as adders_reverb
from acme.agents.jax import actor_core as actor_core_lib
from acme.agents.jax import actors
from acme.agents.jax import builders
from acme.agents.jax.td3 import config as td3_config
# from acme.agents.jax.td3 import learning
import contrastive.learning_td3 as learning
# from acme.agents.jax.td3 import networks as td3_networks
from contrastive import networks_td3 as td3_networks ###@@@###
from acme.datasets import reverb as datasets
from acme.jax import networks as networks_lib
from acme.jax import utils
from acme.jax import variable_utils
from acme.utils import counting
from acme.utils import loggers
import jax
import optax
import reverb
from reverb import rate_limiters

from contrastive import utils as contrastive_utils ###@@@###
from contrastive.episode_saver_adder import EpisodeAdderSaver
import tensorflow as tf
from acme import types
import tree

from contrastive.episode_saver_adder import EpisodeAdderSaver


class ContrastiveBuilderGoalsTD3(builders.ActorLearnerBuilder):
  """TD3 Builder."""

  def __init__(
      self,
      config,
      logger_fn = lambda: None,
      save_data=False,
      data_save_dir=None,
  ):
    """Creates a TD3 learner, a behavior policy and an eval actor.
    Args:
      config: a config with TD3 hps
    """
    self._config = config
    self._logger_fn = logger_fn ###@@@###
    self._save_data = save_data
    self._data_save_dir = data_save_dir ###---###

  def make_learner(
      self,
      random_key,
      networks,
      dataset,
      val_dataset,
      replay_client = None,
      counter = None,
      expert_goals=None, ###===### ###---###
  ):
    critic_optimizer = optax.adam(self._config.critic_learning_rate)
    twin_critic_optimizer = optax.adam(self._config.critic_learning_rate)
    policy_optimizer = optax.adam(self._config.policy_learning_rate)

    if self._config.policy_gradient_clipping is not None:
      policy_optimizer = optax.chain(
          optax.clip_by_global_norm(self._config.policy_gradient_clipping),
          policy_optimizer)

    r_optimizer = optax.adam(self._config.reward_learning_rate)

    return learning.TD3Learner( ###@@@###
        networks=networks,
        random_key=random_key,
        discount=self._config.discount,
        target_sigma=self._config.target_sigma,
        noise_clip=self._config.noise_clip,
        policy_optimizer=policy_optimizer,
        critic_optimizer=critic_optimizer,
        twin_critic_optimizer=twin_critic_optimizer,
        r_optimizer=r_optimizer,
        num_sgd_steps_per_step=self._config.num_sgd_steps_per_step,
        use_sarsa_target=self._config.use_sarsa,
        bc_alpha=self._config.bc_alpha,
        iterator=dataset,
        val_iterator=val_dataset,
        logger=self._logger_fn(),
        config=self._config,
        counter=counter,
        expert_goals=expert_goals)

  def make_actor(
      self,
      random_key,
      policy_network,
      adder = None,
      variable_source = None):
    assert variable_source is not None
    actor_core = actor_core_lib.batched_feed_forward_to_actor_core(
        policy_network)
    variable_client = variable_utils.VariableClient(variable_source, 'policy',
                                                    device='cpu')
    if self._config.use_random_actor:
      # ACTOR = contrastive_utils.InitiallyRandomActor  # pylint: disable=invalid-name
      # ACTOR = contrastive_utils.InitiallyRandomNoGoalActor
      ACTOR = contrastive_utils.InitiallyRandomZeroGoalActor
    else:
      # ACTOR = actors.GenericActor  # pylint: disable=invalid-name
      # ACTOR = contrastive_utils.NoGoalActor
      ACTOR = contrastive_utils.ZeroGoalActor
    return ACTOR(actor_core, random_key, variable_client, adder, obs_dim=self._config.obs_dim, backend='cpu', jit=True)

  def make_replay_tables(
      self,
      environment_spec,
      n_episodes=None,
  ):
    """Create tables to insert data into."""
    samples_per_insert_tolerance = (
        self._config.samples_per_insert_tolerance_rate
        * self._config.samples_per_insert)

    if n_episodes is None:
        min_replay_traj = self._config.min_replay_size // self._config.max_episode_steps  # pylint: disable=line-too-long
        max_replay_traj = self._config.max_replay_size // self._config.max_episode_steps  # pylint: disable=line-too-long
    else:
        min_replay_traj = self._config.min_replay_size // self._config.max_episode_steps
        max_replay_traj = n_episodes

        min_replay_traj += 100
        max_replay_traj += 100
    # min_replay_traj = self._config.min_replay_size // self._config.max_episode_steps  # pylint: disable=line-too-long
    # max_replay_traj = self._config.max_replay_size // self._config.max_episode_steps  # pylint: disable=line-too-long

    print("\nmin_replay_traj:", min_replay_traj)
    print("max_replay_traj:", max_replay_traj, "\n")

    error_buffer = min_replay_traj * samples_per_insert_tolerance
    limiter = rate_limiters.SampleToInsertRatio(
        min_size_to_sample=min_replay_traj,
        samples_per_insert=self._config.samples_per_insert,
        error_buffer=error_buffer)

    return [
        reverb.Table(
            name=self._config.replay_table_name,
            sampler=reverb.selectors.Uniform(),
            remover=reverb.selectors.Fifo(),
            max_size=max_replay_traj,
            rate_limiter=limiter,
            signature=EpisodeAdderSaver.signature(environment_spec, {}))  # pylint: disable=line-too-long
    ]

  def make_dataset_iterator(
      self, replay_client):
    """Create a dataset iterator to use for learning/updating the agent."""
    @tf.function
    def flatten_fn(sample):
      seq_len = tf.shape(sample.data.observation)[0]
      arange = tf.range(seq_len)
      is_future_mask = tf.cast(arange[:, None] < arange[None], tf.float32)
      discount = self._config.discount ** tf.cast(arange[None] - arange[:, None], tf.float32)  # pylint: disable=line-too-long
      probs = is_future_mask * discount
      # The indexing changes the shape from [seq_len, 1] to [seq_len]
      goal_index = tf.random.categorical(logits=tf.math.log(probs),
                                         num_samples=1)[:, 0]
      state = sample.data.observation[:-1, :self._config.obs_dim]
      next_state = sample.data.observation[1:, :self._config.obs_dim]

      # Create the goal observations in three steps.
      # 1. Take all future states (not future goals).
      # 2. Apply obs_to_goal.
      # 3. Sample one of the future states. Note that we don't look for a goal
      # for the final state, because there are no future states.
      goal = sample.data.observation[:, :self._config.obs_dim]
      goal = contrastive_utils.obs_to_goal_2d(
          goal, start_index=self._config.start_index,
          end_index=self._config.end_index)
      goal = tf.gather(goal, goal_index[:-1])
      new_obs = tf.concat([state, goal], axis=1)
      new_next_obs = tf.concat([next_state, goal], axis=1)
      transition = types.Transition(
          observation=new_obs,
          action=sample.data.action[:-1],
          reward=sample.data.reward[:-1],
          discount=sample.data.discount[:-1],
          next_observation=new_next_obs,
          extras={
              'next_action': sample.data.action[1:],
          })
      # Shift for the transpose_shuffle.
      shift = tf.random.uniform((), 0, seq_len, tf.int32)
      transition = tree.map_structure(lambda t: tf.roll(t, shift, axis=0),
                                      transition)
      return transition

    if self._config.num_parallel_calls:
      num_parallel_calls = self._config.num_parallel_calls
    else:
      num_parallel_calls = tf.data.AUTOTUNE

    def _make_dataset(unused_idx):
      dataset = reverb.TrajectoryDataset.from_table_signature(
          server_address=replay_client.server_address,
          table=self._config.replay_table_name,
          max_in_flight_samples_per_worker=100)
      dataset = dataset.map(flatten_fn)
      # transpose_shuffle
      def _transpose_fn(t):
        dims = tf.range(tf.shape(tf.shape(t))[0])
        perm = tf.concat([[1, 0], dims[2:]], axis=0)
        return tf.transpose(t, perm)
      dataset = dataset.batch(self._config.batch_size, drop_remainder=True)
      dataset = dataset.map(
          lambda transition: tree.map_structure(_transpose_fn, transition))
      dataset = dataset.unbatch()
      # end transpose_shuffle

      dataset = dataset.unbatch()
      return dataset
    dataset = tf.data.Dataset.from_tensors(0).repeat()
    dataset = dataset.interleave(
        map_func=_make_dataset,
        cycle_length=num_parallel_calls,
        num_parallel_calls=num_parallel_calls,
        deterministic=False)

    dataset = dataset.batch(
        self._config.batch_size * self._config.num_sgd_steps_per_step,
        drop_remainder=True)
    @tf.function
    def add_info_fn(data):
      info = reverb.SampleInfo(key=0,
                               probability=0.0,
                               table_size=0,
                               priority=0.0)
      return reverb.ReplaySample(info=info, data=data)
    dataset = dataset.map(add_info_fn, num_parallel_calls=tf.data.AUTOTUNE,
                          deterministic=False)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset.as_numpy_iterator()

  def make_adder(self,
                 replay_client,
                 force_no_save=False):
    """Create an adder to record data generated by the actor/environment."""
    return EpisodeAdderSaver(
        client=replay_client,
        priority_fns={self._config.replay_table_name: None},
        max_sequence_length=self._config.max_episode_steps + 1,
        save=False if force_no_save else self._save_data,
        savedir=self._data_save_dir)
