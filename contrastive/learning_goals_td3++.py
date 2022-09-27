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
from contrastive import config_goals as contrastive_config
from contrastive import networks as contrastive_networks
import jax
import jax.numpy as jnp
import optax
import reverb

from jax.experimental import host_callback as hcb
from contrastive.default_logger import make_default_logger


# class TrainingState(NamedTuple):
#   """Contains training state for the learner."""
#   policy_optimizer_state: optax.OptState
#   q_optimizer_state: optax.OptState
#   policy_params: networks_lib.Params
#   q_params: networks_lib.Params
#   target_q_params: networks_lib.Params
#   key: networks_lib.PRNGKey
#   alpha_optimizer_state: Optional[optax.OptState] = None
#   alpha_params: Optional[networks_lib.Params] = None

class TrainingState(NamedTuple):
  """Contains training state for the learner."""
  policy_params: networks_lib.Params
  target_policy_params: networks_lib.Params
  critic_params: networks_lib.Params
  target_critic_params: networks_lib.Params
  twin_critic_params: networks_lib.Params
  target_twin_critic_params: networks_lib.Params
  policy_opt_state: optax.OptState
  critic_opt_state: optax.OptState
  twin_critic_opt_state: optax.OptState
  steps: int
  random_key: networks_lib.PRNGKey


class ContrastiveLearnerGoalsTD3(acme.Learner):
  """Contrastive RL learner."""

  _state: TrainingState



# twin_critic_optimizer
# delay
# target_sigma
# noise_clip
# use_sarsa_target
# bc_alpha

  def __init__(
      self,
      networks,
      random_key # rng,
      policy_optimizer,
      q_optimizer,
      r_optimizer,
      iterator,
      val_iterator,
      counter,
      logger,
      obs_to_goal,
      config,
      expert_goals): ###===### ###---###
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

    # import os
    # # os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    # os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = "0.1"

    expert_goals = jnp.array(expert_goals, dtype=jnp.float32)

    # if config.add_mc_to_td:
    #   assert config.use_td
    # adaptive_entropy_coefficient = config.entropy_coefficient is None
    # self._num_sgd_steps_per_step = config.num_sgd_steps_per_step
    # self._obs_dim = config.obs_dim
    # self._use_td = config.use_td
    #
    # print("\nadaptive_entropy_coefficient:", adaptive_entropy_coefficient)
    # print("self._num_sgd_steps_per_step:", self._num_sgd_steps_per_step)
    # print()
    #
    # if adaptive_entropy_coefficient:
    #   # alpha is the temperature parameter that determines the relative
    #   # importance of the entropy term versus the reward.
    #   log_alpha = jnp.asarray(0., dtype=jnp.float32)
    #   alpha_optimizer = optax.adam(learning_rate=3e-4)
    #   alpha_optimizer_state = alpha_optimizer.init(log_alpha)
    # else:
    #   if config.target_entropy:
    #     raise ValueError('target_entropy should not be set when '
    #                      'entropy_coefficient is provided')

    # def alpha_loss(log_alpha,
    #                policy_params,
    #                transitions,
    #                key):
    #   """Eq 18 from https://arxiv.org/pdf/1812.05905.pdf."""
    #   dist_params = networks.policy_network.apply(
    #       policy_params, transitions.observation)
    #   action = networks.sample(dist_params, key)
    #   log_prob = networks.log_prob(dist_params, action)
    #   alpha = jnp.exp(log_alpha)
    #   alpha_loss = alpha * jax.lax.stop_gradient(
    #       -log_prob - config.target_entropy)
    #   return jnp.mean(alpha_loss)

    def critic_loss(
        critic_params: networks_lib.Params,
        state: TrainingState,
        transition: types.Transition,
        random_key: jnp.ndarray,
    ):
      # Computes the critic loss.
      q_tm1 = networks.critic_network.apply(
          critic_params, transition.observation, transition.action)

      if use_sarsa_target:
        # TODO(b/222674779): use N-steps Trajectories to get the next actions.
        assert 'next_action' in transition.extras, (
            'next actions should be given as extras for one step RL.')
        action = transition.extras['next_action']
      else:
        action = networks.policy_network.apply(state.target_policy_params,
                                               transition.next_observation)
        action = networks.add_policy_noise(action, random_key,
                                           target_sigma, noise_clip)

      q_t = networks.critic_network.apply(
          state.target_critic_params,
          transition.next_observation,
          action)
      twin_q_t = networks.twin_critic_network.apply(
          state.target_twin_critic_params,
          transition.next_observation,
          action)

      q_t = jnp.minimum(q_t, twin_q_t)

      target_q_tm1 = transition.reward + discount * transition.discount * q_t
      td_error = jax.lax.stop_gradient(target_q_tm1) - q_tm1

      return jnp.mean(jnp.square(td_error)), {}


    def reward_loss(r_params,
                    transitions,
                    key,
                    expert_goals):
      batch_size = transitions.observation.shape[0]

      # Negative = policy, positive = expert goals

      s, g = jnp.split(transitions.observation, [config.obs_dim], axis=1)

      idxs = jax.random.randint(key, (batch_size,), 0, expert_goals.shape[0])
      obs_and_goals = jnp.concatenate([s, expert_goals[idxs]], axis=0)
      logits = networks.r_network.apply(r_params, obs_and_goals)[:, 0]
      # (49, 49) = (batch_size, batch_size)
      labels_neg = jnp.zeros(batch_size)
      labels_pos = jnp.ones(batch_size)
      labels = jnp.concatenate([labels_neg, labels_pos], axis=0)
      # labels = jnp.expand_dims(labels, axis=-1)



      # labels_neg = jnp.stack([jnp.ones(batch_size), jnp.zeros(batch_size)], axis=1)
      # labels_pos = jnp.stack([jnp.zeros(batch_size), jnp.ones(batch_size)], axis=1)
      # labels = jnp.concatenate([labels_neg, labels_pos], axis=0)

      if config.reward_loss_type == "bce":
          loss = optax.sigmoid_binary_cross_entropy(logits=logits, labels=labels)
      elif config.reward_loss_type == "pu":
          loss = sigmoid_positive_unlabeled_loss(logits=logits, labels=labels)
      else:
          raise ValueError(f"Unsupported loss type, config.reward_loss_type: {config.reward_loss_type}")

      # loss = jnp.mean(loss)
      # assert len(logits.shape) == 2
      # correct = jnp.argmax(logits, axis=1) == jnp.argmax(labels, axis=1)
      # logits_neg = logits[:batch_size, 0]
      # logits_pos = logits[batch_size:, 0]
      # logsumexp = jax.nn.logsumexp(logits, axis=1)**2
      #
      # metrics = {
      #     'binary_accuracy': jnp.mean((logits > 0) == labels),
      #     'categorical_accuracy': jnp.mean(correct),
      #     'logits_pos': logits_pos,
      #     'logits_neg': logits_neg,
      #     'logsumexp': logsumexp.mean(),
      # }

      loss = jnp.mean(loss)
      assert len(logits.shape) == 1
      # correct = jnp.squeeze(logits) == jnp.squeeze(labels)
      logits_neg = logits[:batch_size].mean()
      logits_pos = logits[batch_size:].mean()
      # logsumexp = jax.nn.logsumexp(logits, axis=1)**2

      sigmoid_neg = jax.nn.sigmoid(logits[:batch_size]).mean()
      sigmoid_pos = jax.nn.sigmoid(logits[batch_size:]).mean()

      metrics = {
          'binary_accuracy': jnp.mean((logits > 0) == labels),
          'sigmoid_binary_accuracy': jnp.mean((jax.nn.sigmoid(logits) > 0.5) == labels),
          # 'categorical_accuracy': jnp.mean(correct),
          'logits_pos': logits_pos,
          'logits_neg': logits_neg,
          'sigmoid_pos': sigmoid_pos,
          'sigmoid_neg': sigmoid_neg,
          # 'logsumexp': logsumexp.mean(),
      }

      return loss, metrics


    def policy_loss(
        policy_params: networks_lib.Params,
        critic_params: networks_lib.Params,
        transition: types.NestedArray,
    ) -> jnp.ndarray:
      # Computes the discrete policy gradient loss.
      action = networks.policy_network.apply(
          policy_params, transition.observation)
      grad_critic = jax.vmap(
          jax.grad(networks.critic_network.apply, argnums=2),
          in_axes=(None, 0, 0))
      dq_da = grad_critic(critic_params, transition.observation, action)
      batch_dpg_learning = jax.vmap(rlax.dpg_loss, in_axes=(0, 0))
      loss = jnp.mean(batch_dpg_learning(action, dq_da))
      if bc_alpha is not None:
        # BC regularization for offline RL
        q_sa = networks.critic_network.apply(critic_params,
                                             transition.observation, action)
        bc_factor = jax.lax.stop_gradient(bc_alpha / jnp.mean(jnp.abs(q_sa)))
        loss += jnp.mean(jnp.square(action - transition.action)) / bc_factor
      return loss


    alpha_grad = jax.value_and_grad(alpha_loss)
    critic_grad = jax.value_and_grad(critic_loss, has_aux=True)
    actor_grad = jax.value_and_grad(actor_loss)
    reward_grad = jax.value_and_grad(reward_loss, has_aux=True)

    def update_step(
        state,
        all_transitions,
    ):

      transitions, val_transitions = all_transitions
      key, key_alpha, key_critic, key_actor, key_reward = jax.random.split(state.key, 5)

      if adaptive_entropy_coefficient:
        alpha_loss, alpha_grads = alpha_grad(state.alpha_params,
                                             state.policy_params, transitions,
                                             key_alpha)
        alpha = jnp.exp(state.alpha_params)
      else:
        alpha = config.entropy_coefficient

      if not config.use_gcbc:
        (critic_loss, critic_metrics), critic_grads = critic_grad(
            state.q_params, state.policy_params, state.target_q_params, state.r_params,
            transitions, key_critic)

      actor_loss, actor_grads = actor_grad(state.policy_params, state.q_params,
                                           alpha, transitions, key_actor, expert_goals)

      # Apply policy gradients
      actor_update, policy_optimizer_state = policy_optimizer.update(
          actor_grads, state.policy_optimizer_state)
      policy_params = optax.apply_updates(state.policy_params, actor_update)

      if config.use_td:
          (val_reward_loss, val_reward_metrics), _ = reward_grad(
              state.r_params, val_transitions, key_reward, expert_goals)

          (reward_loss, reward_metrics), reward_grads = reward_grad(
              state.r_params, transitions, key_reward, expert_goals)

          # Apply reward gradients
          reward_update, r_optimizer_state = r_optimizer.update(
              reward_grads, state.r_optimizer_state)
          r_params = optax.apply_updates(state.r_params, reward_update)
      else:
          val_reward_loss = reward_loss = 0
          val_reward_metrics = reward_metrics = {}

      # Apply critic gradients
      if config.use_gcbc:
        metrics = {}
        critic_loss = 0.0
        q_params = state.q_params
        q_optimizer_state = state.q_optimizer_state
        new_target_q_params = state.target_q_params
      else:
        critic_update, q_optimizer_state = q_optimizer.update(
            critic_grads, state.q_optimizer_state)

        q_params = optax.apply_updates(state.q_params, critic_update)

        new_target_q_params = jax.tree_map(
            lambda x, y: x * (1 - config.tau) + y * config.tau,
            state.target_q_params, q_params)
        metrics = critic_metrics

      metrics.update({
          'critic_loss': critic_loss,
          'actor_loss': actor_loss,
          'reward_loss': reward_loss,
          'val_reward_loss': val_reward_loss,
      })

      metrics.update({"train_" + key:val for key, val in reward_metrics.items()})
      metrics.update({"val_" + key:val for key, val in val_reward_metrics.items()})

      new_state = TrainingState(
          policy_optimizer_state=policy_optimizer_state,
          q_optimizer_state=q_optimizer_state,
          r_optimizer_state=r_optimizer_state,
          policy_params=policy_params,
          q_params=q_params,
          r_params=r_params,
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
    self._val_iterator = val_iterator

    update_step = utils.process_multiple_batches(update_step,
                                                 config.num_sgd_steps_per_step)
    # Use the JIT compiler.
    if config.jit:
      self._update_step = jax.jit(update_step)
    else:
      self._update_step = update_step

    def make_initial_state(key):
      """Initialises the training state (parameters and optimiser state)."""
      key_policy, key_q, key_r, key = jax.random.split(key, 4)

      policy_params = networks.policy_network.init(key_policy)
      policy_optimizer_state = policy_optimizer.init(policy_params)

      q_params = networks.q_network.init(key_q)
      q_optimizer_state = q_optimizer.init(q_params)

      r_params = networks.r_network.init(key_r)
      r_optimizer_state = r_optimizer.init(r_params)

      state = TrainingState(
          policy_optimizer_state=policy_optimizer_state,
          q_optimizer_state=q_optimizer_state,
          r_optimizer_state=r_optimizer_state,
          policy_params=policy_params,
          q_params=q_params,
          r_params=r_params,
          target_q_params=q_params,
          key=key)

      if adaptive_entropy_coefficient:
        state = state._replace(alpha_optimizer_state=alpha_optimizer_state,
                               alpha_params=log_alpha)
      return state

    # Create initial state.
    self._state = make_initial_state(rng)

    # Do not record timestamps until after the first learning step is done.
    # This is to avoid including the time it takes for actors to come online
    # and fill the replay buffer.
    self._timestamp = None

  def step(self):
    with jax.profiler.StepTraceAnnotation('step', step_num=self._counter):
      sample = next(self._iterator)
      transitions = types.Transition(*sample.data)

      val_sample = next(self._val_iterator)
      val_transitions = types.Transition(*val_sample.data)

      # self._state, metrics = self._update_step(self._state, (transitions, val_transitions))
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
        'twin_critic': self._state.twin_critic_params,
        'reward': self._state.r_params,
    }
    return [variables[name] for name in names]
