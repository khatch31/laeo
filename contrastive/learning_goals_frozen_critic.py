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

"""Contrastive RL learner implementation."""
import time
from typing import Any, Dict, Iterator, List, NamedTuple, Optional, Tuple, Callable

import acme
from acme import types
from acme.jax import networks as networks_lib
from acme.jax import utils
from acme.utils import counting
from acme.utils import loggers
from contrastive import config_goals_frozen_critic as contrastive_config
from contrastive import networks as contrastive_networks
import jax
import jax.numpy as jnp
import optax
import reverb

from jax.experimental import host_callback as hcb
from contrastive.default_logger import make_default_logger

class TrainingState(NamedTuple):
  """Contains training state for the learner."""
  policy_optimizer_state: optax.OptState
  q_optimizer_state: optax.OptState
  policy_params: networks_lib.Params
  q_params: networks_lib.Params
  target_q_params: networks_lib.Params
  key: networks_lib.PRNGKey
  alpha_optimizer_state: Optional[optax.OptState] = None
  alpha_params: Optional[networks_lib.Params] = None


class ContrastiveLearnerGoalsFrozenCritic(acme.Learner):
  """Contrastive RL learner."""

  _state: TrainingState

  def __init__(
      self,
      networks,
      rng,
      policy_optimizer,
      q_optimizer,
      iterator,
      counter,
      logger,
      obs_to_goal,
      config,
      expert_goals,
      critic_checkpoint_state,): ###===### ###---###
    """Initialize the Contrastive RL learner.

    Args:
      networks: Contrastive RL networks.
      rng: a key for random number generation.
      policy_optimizer: the policy optimizer.
      q_optimizer: the Q-function optimizer.
      iterator: an iterator over training data.
      counter: counter object used to keep track of steps.
      logger: logger object to be used by learner.
      obs_to_goal: a function for extracting the goal coordinates.
      config: the experiment config file.
    """

    expert_goals = jnp.array(expert_goals, dtype=jnp.float32)

    if config.add_mc_to_td:
      assert config.use_td
    adaptive_entropy_coefficient = config.entropy_coefficient is None
    self._num_sgd_steps_per_step = config.num_sgd_steps_per_step
    self._obs_dim = config.obs_dim
    self._use_td = config.use_td

    print("adaptive_entropy_coefficient:", adaptive_entropy_coefficient)
    
    if adaptive_entropy_coefficient:
      # alpha is the temperature parameter that determines the relative
      # importance of the entropy term versus the reward.
      log_alpha = jnp.asarray(0., dtype=jnp.float32)
      alpha_optimizer = optax.adam(learning_rate=3e-4)
      alpha_optimizer_state = alpha_optimizer.init(log_alpha)
    else:
      if config.target_entropy:
        raise ValueError('target_entropy should not be set when '
                         'entropy_coefficient is provided')

    def alpha_loss(log_alpha,
                   policy_params,
                   transitions,
                   key):
      """Eq 18 from https://arxiv.org/pdf/1812.05905.pdf."""
      dist_params = networks.policy_network.apply(
          policy_params, transitions.observation)
      action = networks.sample(dist_params, key)
      log_prob = networks.log_prob(dist_params, action)
      alpha = jnp.exp(log_alpha)
      alpha_loss = alpha * jax.lax.stop_gradient(
          -log_prob - config.target_entropy)
      return jnp.mean(alpha_loss)


    def actor_loss(policy_params,
                   q_params,
                   alpha,
                   transitions,
                   key,
                   expert_goals,
                   ):
      obs = transitions.observation
      if config.use_gcbc: # Just does behavioral cloning here?
        dist_params = networks.policy_network.apply(
            policy_params, obs)
        log_prob = networks.log_prob(dist_params, transitions.action)
        actor_loss = -1.0 * jnp.mean(log_prob)
      else:
        state = obs[:, :config.obs_dim]
        # goal = obs[:, config.obs_dim:]
        actor_goal = jnp.zeros_like(obs[:, config.obs_dim:])

        batch_size = obs.shape[0]
        idxs = jax.random.randint(key, (batch_size,), 0, expert_goals.shape[0])
        critic_goal = expert_goals[idxs]
        # hcb.id_print(actor_goal, what="\n\nactor_goal")
        # hcb.id_print(actor_goal.shape, what="actor_goal.shape")
        # hcb.id_print(idxs, what="idxs")
        # hcb.id_print(critic_goal, what="critic_goal")
        # hcb.id_print(critic_goal.shape, what="critic_goal.shape")

        if config.random_goals == 0.0:
          new_state = state
          # new_goal = goal
          new_actor_goal = actor_goal
          new_critic_goal = critic_goal
        elif config.random_goals == 0.5:
          new_state = jnp.concatenate([state, state], axis=0)
          # new_goal = jnp.concatenate([goal, jnp.roll(goal, 1, axis=0)], axis=0)
          new_actor_goal = jnp.concatenate([actor_goal, jnp.roll(actor_goal, 1, axis=0)], axis=0)
          new_critic_goal = jnp.concatenate([critic_goal, jnp.roll(critic_goal, 1, axis=0)], axis=0)
        else:
          assert config.random_goals == 1.0
          new_state = state
          # new_goal = jnp.roll(goal, 1, axis=0)
          new_actor_goal = jnp.roll(actor_goal, 1, axis=0)
          new_critic_goal = jnp.roll(critic_goal, 1, axis=0)

        # new_obs = jnp.concatenate([new_state, new_goal], axis=1)
        new_actor_obs = jnp.concatenate([new_state, new_actor_goal], axis=1)
        # hcb.id_print(new_actor_obs, what="new_actor_obs")
        # hcb.id_print(new_actor_obs.shape, what="new_actor_obs.shape")
        dist_params = networks.policy_network.apply(
            policy_params, new_actor_obs)
        action = networks.sample(dist_params, key)
        log_prob = networks.log_prob(dist_params, action)

        new_critic_obs = jnp.concatenate([new_state, new_critic_goal], axis=1)
        # hcb.id_print(new_critic_obs, what="new_critic_obs")
        # hcb.id_print(new_critic_obs.shape, what="new_critic_obs.shape")
        q_action = networks.q_network.apply(
            q_params, new_critic_obs, action)
        if len(q_action.shape) == 3:  # twin q trick
          assert q_action.shape[2] == 2
          q_action = jnp.min(q_action, axis=-1)
        actor_loss = alpha * log_prob - jnp.diag(q_action)

      return jnp.mean(actor_loss)

    alpha_grad = jax.value_and_grad(alpha_loss)
    actor_grad = jax.value_and_grad(actor_loss)

    def update_step(
        state,
        transitions,
    ):

      key, key_alpha, key_critic, key_actor = jax.random.split(state.key, 4)
      if adaptive_entropy_coefficient:
        alpha_loss, alpha_grads = alpha_grad(state.alpha_params,
                                             state.policy_params, transitions,
                                             key_alpha)
        alpha = jnp.exp(state.alpha_params)
      else:
        alpha = config.entropy_coefficient

      actor_loss, actor_grads = actor_grad(state.policy_params, state.q_params,
                                           alpha, transitions, key_actor, expert_goals)

      # Apply policy gradients
      actor_update, policy_optimizer_state = policy_optimizer.update(
          actor_grads, state.policy_optimizer_state)
      policy_params = optax.apply_updates(state.policy_params, actor_update)

      # Apply critic gradients
      metrics = {}
      critic_loss = 0.0
      q_params = state.q_params
      q_optimizer_state = state.q_optimizer_state
      new_target_q_params = state.target_q_params

      metrics.update({
          'critic_loss': critic_loss,
          'actor_loss': actor_loss,
      })

      new_state = TrainingState(
          policy_optimizer_state=policy_optimizer_state,
          q_optimizer_state=q_optimizer_state,
          policy_params=policy_params,
          q_params=q_params,
          target_q_params=new_target_q_params,
          key=key,
      )
      if adaptive_entropy_coefficient:
        # Apply alpha gradients
        alpha_update, alpha_optimizer_state = alpha_optimizer.update(
            alpha_grads, state.alpha_optimizer_state)
        alpha_params = optax.apply_updates(state.alpha_params, alpha_update)
        metrics.update({
            'alpha_loss': alpha_loss,
            'alpha': jnp.exp(alpha_params),
        })
        new_state = new_state._replace(
            alpha_optimizer_state=alpha_optimizer_state,
            alpha_params=alpha_params)

      return new_state, metrics

    # General learner book-keeping and loggers.
    self._counter = counter or counting.Counter()
    self._logger = logger or make_default_logger(
        "~/acme",
        'learner', asynchronous=True, serialize_fn=utils.fetch_devicearray,
        time_delta=10.0,
        wandblogger=None)

    # Iterator on demonstration transitions.
    self._iterator = iterator

    update_step = utils.process_multiple_batches(update_step,
                                                 config.num_sgd_steps_per_step)
    # Use the JIT compiler.
    if config.jit:
      self._update_step = jax.jit(update_step)
    else:
      self._update_step = update_step

    def make_initial_state(key):
      """Initialises the training state (parameters and optimiser state)."""
      key_policy, key_q, key = jax.random.split(key, 3)

      policy_params = networks.policy_network.init(key_policy)
      policy_optimizer_state = policy_optimizer.init(policy_params)

      q_params = networks.q_network.init(key_q)
      q_optimizer_state = q_optimizer.init(q_params)

      state = TrainingState(
          policy_optimizer_state=policy_optimizer_state,
          q_optimizer_state=q_optimizer_state,
          policy_params=policy_params,
          q_params=q_params,
          target_q_params=q_params,
          key=key)

      if adaptive_entropy_coefficient:
        state = state._replace(alpha_optimizer_state=alpha_optimizer_state,
                               alpha_params=log_alpha)
      return state

    # Create initial state.
    self._state = make_initial_state(rng)

    assert critic_checkpoint_state is not None
    if critic_checkpoint_state is not None:
        self.restore_critic_only(critic_checkpoint_state)

    # Do not record timestamps until after the first learning step is done.
    # This is to avoid including the time it takes for actors to come online
    # and fill the replay buffer.
    self._timestamp = None

  def step(self):
    with jax.profiler.StepTraceAnnotation('step', step_num=self._counter):
      sample = next(self._iterator)
      transitions = types.Transition(*sample.data)
      self._state, metrics = self._update_step(self._state, transitions)

    # Compute elapsed time.
    timestamp = time.time()
    elapsed_time = timestamp - self._timestamp if self._timestamp else 0
    self._timestamp = timestamp

    # Increment counts and record the current time
    counts = self._counter.increment(steps=1, walltime=elapsed_time)
    if elapsed_time > 0:
      metrics['steps_per_second'] = (
          self._num_sgd_steps_per_step / elapsed_time)
    else:
      metrics['steps_per_second'] = 0.

    # Attempts to write the logs.
    self._logger.write({**metrics, **counts})


  def get_variables(self, names):
    variables = {
        'policy': self._state.policy_params,
        'critic': self._state.q_params,
    }
    return [variables[name] for name in names]

  def save(self):
    return self._state

  def restore(self, state):
    self._state = state

  def restore_critic_only(self, state):
      new_state = TrainingState(
          policy_optimizer_state=self._state.policy_optimizer_state,
          q_optimizer_state=state.q_optimizer_state,
          policy_params=self._state.policy_params,
          q_params=state.q_params,
          target_q_params=state.target_q_params,
          key=self._state.key,
      )
      self._state = new_state
