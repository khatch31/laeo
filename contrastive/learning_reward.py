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
from contrastive import config_reward as contrastive_config
from contrastive import networks as contrastive_networks
import jax
import jax.numpy as jnp
import optax
import reverb

from contrastive.losses import sigmoid_positive_unlabeled_loss


from jax.experimental import host_callback as hcb
from contrastive.default_logger import make_default_logger


class TrainingState2(NamedTuple):
  """Contains training state for the learner."""
  policy_optimizer_state: optax.OptState
  q_optimizer_state: optax.OptState
  r_optimizer_state: optax.OptState
  policy_params: networks_lib.Params
  q_params: networks_lib.Params
  r_params: networks_lib.Params
  target_q_params: networks_lib.Params
  key: networks_lib.PRNGKey
  alpha_optimizer_state: Optional[optax.OptState] = None
  alpha_params: Optional[networks_lib.Params] = None


class ContrastiveLearnerReward(acme.Learner):
  """Contrastive RL learner."""

  _state: TrainingState2

  def __init__(
      self,
      networks,
      rng,
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

    if config.add_mc_to_td:
      assert config.use_td
    adaptive_entropy_coefficient = config.entropy_coefficient is None
    self._num_sgd_steps_per_step = config.num_sgd_steps_per_step
    self._obs_dim = config.obs_dim
    self._use_td = config.use_td

    print("\nadaptive_entropy_coefficient:", adaptive_entropy_coefficient)
    print("self._num_sgd_steps_per_step:", self._num_sgd_steps_per_step)
    print("config.jit:", config.jit)
    print()

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


    def critic_loss(q_params,
                    policy_params,
                    target_q_params,
                    transitions,
                    key):
      batch_size = transitions.observation.shape[0]
      # Note: We might be able to speed up the computation for some of the
      # baselines to making a single network that returns all the values. This
      # avoids computing some of the underlying representations multiple times.
      if config.use_td:
        # For TD learning, the diagonal elements are the immediate next state.
        s, g = jnp.split(transitions.observation, [config.obs_dim], axis=1)
        next_s, _ = jnp.split(transitions.next_observation, [config.obs_dim],
                              axis=1)
        if config.add_mc_to_td:
          next_fraction = (1 - config.discount) / ((1 - config.discount) + 1)
          num_next = int(batch_size * next_fraction)
          new_g = jnp.concatenate([
              obs_to_goal(next_s[:num_next]),
              g[num_next:],
          ], axis=0)
        else:
          new_g = obs_to_goal(next_s)
        obs = jnp.concatenate([s, new_g], axis=1)
        transitions = transitions._replace(observation=obs)

      I = jnp.eye(batch_size)  # pylint: disable=invalid-name
      # hcb.id_print(transitions.observation, what="\n\transitions.observation")
      # hcb.id_print(transitions.observation.shape, what="transitions.observation.shape")
      # logits = networks.q_network.apply(
      #     q_params, transitions.observation, transitions.action)
      logits = networks.q_network.apply(
          q_params, transitions.observation, transitions.action) ###%%%###


      if config.use_td:
        # Make sure to use the twin Q trick.
        assert len(logits.shape) == 3

        # We evaluate the next-state Q function using random goals
        s, g = jnp.split(transitions.observation, [config.obs_dim], axis=1)
        del s
        next_s = transitions.next_observation[:, :config.obs_dim]
        goal_indices = jnp.roll(jnp.arange(batch_size, dtype=jnp.int32), -1)
        g = g[goal_indices]
        transitions = transitions._replace(
            next_observation=jnp.concatenate([next_s, g], axis=1))
        next_dist_params = networks.policy_network.apply(
            policy_params, transitions.next_observation)
        next_action = networks.sample(next_dist_params, key)
        next_q = networks.q_network.apply(target_q_params,
                                          transitions.next_observation,
                                          next_action)  # This outputs logits.
        next_q = jax.nn.sigmoid(next_q)
        next_v = jnp.min(next_q, axis=-1)
        next_v = jax.lax.stop_gradient(next_v)
        next_v = jnp.diag(next_v)
        # diag(logits) are predictions for future states.
        # diag(next_q) are predictions for random states, which correspond to
        # the predictions logits[range(B), goal_indices].
        # So, the only thing that's meaningful for next_q is the diagonal. Off
        # diagonal entries are meaningless and shouldn't be used.
        w = next_v / (1 - next_v)
        w_clipping = 20.0
        w = jnp.clip(w, 0, w_clipping)
        # (B, B, 2) --> (B, 2), computes diagonal of each twin Q.
        pos_logits = jax.vmap(jnp.diag, -1, -1)(logits)
        loss_pos = optax.sigmoid_binary_cross_entropy(
            logits=pos_logits, labels=1)  # [B, 2]

        neg_logits = logits[jnp.arange(batch_size), goal_indices]
        loss_neg1 = w[:, None] * optax.sigmoid_binary_cross_entropy(
            logits=neg_logits, labels=1)  # [B, 2]
        loss_neg2 = optax.sigmoid_binary_cross_entropy(
            logits=neg_logits, labels=0)  # [B, 2]

        if config.add_mc_to_td:
          loss = ((1 + (1 - config.discount)) * loss_pos
                  + config.discount * loss_neg1 + 2 * loss_neg2)
        else:
          loss = ((1 - config.discount) * loss_pos
                  + config.discount * loss_neg1 + loss_neg2)
        # Take the mean here so that we can compute the accuracy.
        logits = jnp.mean(logits, axis=-1)

      else:  # For the MC losses.
        def loss_fn(_logits):  # pylint: disable=invalid-name
          if config.use_cpc:
            return (optax.softmax_cross_entropy(logits=_logits, labels=I)
                    + 0.01 * jax.nn.logsumexp(_logits, axis=1)**2)
          else:
            return optax.sigmoid_binary_cross_entropy(logits=_logits, labels=I)
        if len(logits.shape) == 3:  # twin q
          # loss.shape = [.., num_q]
          loss = jax.vmap(loss_fn, in_axes=2, out_axes=-1)(logits)
          loss = jnp.mean(loss, axis=-1)
          # Take the mean here so that we can compute the accuracy.
          logits = jnp.mean(logits, axis=-1)
        else:
          loss = loss_fn(logits)

      loss = jnp.mean(loss)
      correct = (jnp.argmax(logits, axis=1) == jnp.argmax(I, axis=1))
      logits_pos = jnp.sum(logits * I) / jnp.sum(I)
      logits_neg = jnp.sum(logits * (1 - I)) / jnp.sum(1 - I)
      if len(logits.shape) == 3:
        logsumexp = jax.nn.logsumexp(logits[:, :, 0], axis=1)**2
      else:
        logsumexp = jax.nn.logsumexp(logits, axis=1)**2
      metrics = {
          'binary_accuracy': jnp.mean((logits > 0) == I),
          'categorical_accuracy': jnp.mean(correct),
          'logits_pos': logits_pos,
          'logits_neg': logits_neg,
          'logsumexp': logsumexp.mean(),
      }

      return loss, metrics

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


    def actor_loss(policy_params,
                   q_params,
                   alpha,
                   transitions,
                   key,
                   expert_goals,
                   ):
      obs = transitions.observation
      # hcb.id_print(obs[0], what="\n\nobs")
      if config.use_gcbc: # Just does behavioral cloning here?
        state = obs[:, :config.obs_dim]
        actor_goal = jnp.zeros_like(obs[:, config.obs_dim:])
        new_actor_obs = jnp.concatenate([state, actor_goal], axis=1)
        # hcb.id_print(new_actor_obs, what="\n\nnew_actor_obs")
        dist_params = networks.policy_network.apply(
            policy_params, new_actor_obs)
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
        # hcb.id_print(new_actor_obs[0], what="new_actor_obs")
        # hcb.id_print(new_actor_obs.shape, what="new_actor_obs.shape")
        dist_params = networks.policy_network.apply(
            policy_params, new_actor_obs)
        action = networks.sample(dist_params, key)
        log_prob = networks.log_prob(dist_params, action)

        new_critic_obs = jnp.concatenate([new_state, new_critic_goal], axis=1)
        # hcb.id_print(new_critic_obs[0], what="new_critic_obs")
        # hcb.id_print(new_critic_obs.shape, what="new_critic_obs.shape")

        # add jnp.exp
        # inverting actor loss?

        q_action = networks.q_network.apply(
            q_params, new_critic_obs, action)
        if len(q_action.shape) == 3:  # twin q trick
          assert q_action.shape[2] == 2
          q_action = jnp.min(q_action, axis=-1)
        if config.exp_q_action:
            hcb.id_print(q_action, what="(before) q_action")
            q_action = jnp.exp(q_action)
            hcb.id_print(q_action, what="(after) q_action")

        actor_loss = alpha * log_prob - jnp.diag(q_action)

        if config.invert_actor_loss:
            # hcb.id_print(actor_loss, what="(before) actor_loss")
            actor_loss = -actor_loss
            # hcb.id_print(actor_loss, what="(after) actor_loss")

        assert 0.0 <= config.bc_coef <= 1.0
        if config.bc_coef > 0:
          orig_action = transitions.action
          if config.random_goals == 0.5:
            orig_action = jnp.concatenate([orig_action, orig_action], axis=0)

          bc_loss = -1.0 * networks.log_prob(dist_params, orig_action)
          actor_loss = (config.bc_coef * bc_loss
                        + (1 - config.bc_coef) * actor_loss)

      return jnp.mean(actor_loss)

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

      (val_reward_loss, val_reward_metrics), _ = reward_grad(
          state.r_params, val_transitions, key_reward, expert_goals)

      (reward_loss, reward_metrics), reward_grads = reward_grad(
          state.r_params, transitions, key_reward, expert_goals)

      # Apply reward gradients
      reward_update, r_optimizer_state = r_optimizer.update(
          reward_grads, state.r_optimizer_state)
      r_params = optax.apply_updates(state.r_params, reward_update)

      metrics = {"train_" + key:val for key, val in reward_metrics.items()}
      metrics.update({"val_" + key:val for key, val in val_reward_metrics.items()})

      critic_loss = 0.0
      q_params = state.q_params
      q_optimizer_state = state.q_optimizer_state
      new_target_q_params = state.target_q_params

      actor_loss = 0.0
      policy_params = state.policy_params
      policy_optimizer_state = state.policy_optimizer_state

      # reward_loss = 0.0
      # r_params = state.r_params
      # r_optimizer_state = state.r_optimizer_state

      metrics.update({
          # 'critic_loss': critic_loss,
          # 'actor_loss': actor_loss,
          'reward_loss': reward_loss,
          'val_reward_loss': val_reward_loss,
      })

      new_state = TrainingState2(
          policy_optimizer_state=policy_optimizer_state,
          q_optimizer_state=q_optimizer_state,
          r_optimizer_state=r_optimizer_state,
          policy_params=policy_params,
          q_params=q_params,
          r_params=r_params,
          target_q_params=new_target_q_params,
          key=key,
      )
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

      state = TrainingState2(
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

      self._state, metrics = self._update_step(self._state, (transitions, val_transitions))

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
        'reward': self._state.r_params,
    }
    return [variables[name] for name in names]

  def save(self):
    return self._state

  def restore(self, state):
    self._state = state
