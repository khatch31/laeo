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

from contrastive.losses import sigmoid_positive_unlabeled_loss

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
  steps: int
  alpha_optimizer_state: Optional[optax.OptState] = None
  alpha_params: Optional[networks_lib.Params] = None


# class TrainingState(NamedTuple):
#   """Contains training state for the learner."""
#   policy_optimizer_state: optax.OptState
#   q_optimizer_state: optax.OptState
#   r_optimizer_state: optax.OptState
#   policy_params: networks_lib.Params
#   q_params: networks_lib.Params
#   r_params: networks_lib.Params
#   target_q_params: networks_lib.Params
#   key: networks_lib.PRNGKey
#   alpha_optimizer_state: Optional[optax.OptState] = None
#   alpha_params: Optional[networks_lib.Params] = None


class ContrastiveLearnerGoals(acme.Learner):
  """Contrastive RL learner."""

  _state: TrainingState

  def __init__(
      self,
      networks,
      rng,
      policy_optimizer,
      q_optimizer,
      # r_optimizer,
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
    print()

    # self.val_critic_loss = 0
    # self.val_critic_metrics = {}
    # self.val_actor_loss = 0

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

      I = jnp.eye(batch_size)  # pylint: disable=invalid-name
      # hcb.id_print(transitions.observation, what="\n\transitions.observation")
      # hcb.id_print(transitions.observation.shape, what="transitions.observation.shape")
      logits = networks.q_network.apply(
          q_params, transitions.observation, transitions.action)

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

    def actor_loss(policy_params,
                   q_params,
                   alpha,
                   transitions,
                   key,
                   expert_goals,
                   ):
      obs = transitions.observation
      # hcb.id_print(obs.shape, what="\n\nobs.shape")
      # hcb.id_print(obs[0], what="\n\nobs")
      if config.use_gcbc: # Just does behavioral cloning here?
        state = obs[:, :config.obs_dim]
        actor_goal = jnp.zeros_like(obs[:, config.obs_dim:])
        new_actor_obs = jnp.concatenate([state, actor_goal], axis=1)
        # hcb.id_print(new_actor_obs, what="\n\nnew_actor_obs")
        # dist_params = networks.policy_network.apply(policy_params, new_actor_obs)
        dist_params = networks.policy_network.apply(policy_params, new_actor_obs[:, :config.obs_dim])

        if config.mse_bc_loss:
            # action = networks.sample(dist_params, key)
            action = dist_params.mode()
            actor_loss = jnp.square(action - transitions.action)
        else:
            log_prob = networks.log_prob(dist_params, transitions.action)
            actor_loss = -1.0 * jnp.mean(log_prob)
        metrics = {}
      else:
        state = obs[:, :config.obs_dim]
        # goal = obs[:, config.obs_dim:]

        if config.revert_to_goal_conditioned:
            actor_goal = obs[:, config.obs_dim:]
            critic_goal = actor_goal.copy()
        else:
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
        # dist_params = networks.policy_network.apply(policy_params, new_actor_obs)

        if config.revert_to_goal_conditioned:
            # hcb.id_print(new_actor_obs, what="\nnew_actor_obs")
            dist_params = networks.policy_network.apply(policy_params, new_actor_obs)
        else:
            dist_params = networks.policy_network.apply(policy_params, new_actor_obs[:, :config.obs_dim])

        action = networks.sample(dist_params, key)
        log_prob = networks.log_prob(dist_params, action)



        if config.max_expert_examples:
            # expert_goals = expert_goals.at[:].set(0) # COMMENT OUT
            # expert_goals = expert_goals.at[13].set(1000) # COMMENT OUT
            # hcb.id_print(expert_goals, what="\nexpert_goals")

            # new_critic_goal = new_critic_goal.at[:-1].set(0)
            # new_critic_goal = new_critic_goal.at[-1].set(1000)
            # hcb.id_print(new_critic_goal, what="\nnew_critic_goal")
            # new_critic_obs = jnp.concatenate([new_state, new_critic_goal], axis=1)
            # qbob = networks.q_network.apply(q_params, new_critic_obs, action)
            # # hcb.id_print(qbob, what="\nqbob")
            # qbob = jnp.min(qbob, axis=-1)
            # hcb.id_print(jnp.diag(qbob), what="\njnp.diag(qbob)")

            expert_goals_tiled = jnp.tile(expert_goals, (batch_size, 1, 1)) # (batch_size, n_examples, obs_dim) = (1024, 50, 5)

            def critic_vmap_fn(n_state, exp_tiled, act):
                new_critic_obs = jnp.concatenate([n_state, exp_tiled], axis=1)
                q_action = networks.q_network.apply(q_params, new_critic_obs, act)
                return q_action

            vmapped_critic = jax.vmap(critic_vmap_fn, (None, 1, None), 2)
            q_action = vmapped_critic(new_state, expert_goals_tiled, action) # (batch_size, batch_size, n_examples, 2) = (1024, 1024, 50, 2)
            # max_idxs = jnp.argmax(jnp.abs(q_action), axis=2)
            q_action = jnp.max(q_action, axis=2) # (batch_size, batch_size, 2) = (1024, 1024, 2) # UNCOMMENT

            # q_action = jnp.max(jnp.abs(q_action), axis=2) # (batch_size, batch_size, 2) = (1024, 1024, 2) # COMMENT OUT

            # hcb.id_print(q_action, what="\nq_action")
            # hcb.id_print(max_idxs, what="\nmax_idxs")
            # hcb.id_print(max_idxs.shape, what="\nmax_idxs.shape")


        elif config.logsumexp_expert_examples:
            expert_goals_tiled = jnp.tile(expert_goals, (batch_size, 1, 1)) # (batch_size, n_examples, obs_dim) = (1024, 50, 5)

            def critic_vmap_fn(n_state, exp_tiled, act):
                new_critic_obs = jnp.concatenate([n_state, exp_tiled], axis=1)
                q_action = networks.q_network.apply(q_params, new_critic_obs, act)
                return q_action

            vmapped_critic = jax.vmap(critic_vmap_fn, (None, 1, None), 2)
            q_action = vmapped_critic(new_state, expert_goals_tiled, action) # (batch_size, batch_size, n_examples, obs_dim) = (1024, 1024, 50, 2)
            q_action = jax.nn.logsumexp(q_action, axis=2) # (batch_size, batch_size, 2) = (1024, 1024, 2)
        elif config.hack_expert_examples:
            # hcb.id_print(new_state[:5, 3], what="\nnew_state[:5, 3]")
            hacked_critic_goal = new_critic_goal.copy()
            # hcb.id_print(hacked_critic_goal[:5, 3], what="(before) hacked_critic_goal[:5, 3]")
            hacked_critic_goal = hacked_critic_goal.at[:, 3].set(new_state[:, 3])
            # hcb.id_print(hacked_critic_goal[:5, 3], what="(after) hacked_critic_goal[:5, 3]")
            # hcb.id_print(new_critic_goal.shape, what="new_critic_goal.shape")
            # hcb.id_print(hacked_critic_goal.shape, what="hacked_critic_goal.shape")
            new_critic_obs = jnp.concatenate([new_state, hacked_critic_goal], axis=1)
            # hcb.id_print(new_state[:2], what="\nnew_state[:2]")
            # hcb.id_print(new_critic_obs[:2], what="new_critic_obs[:2]")
            # hcb.id_print(jnp.concatenate([new_state, new_critic_goal], axis=1)[:2], what="jnp.concatenate([new_state, new_critic_goal], axis=1)[:2]")

            q_action = networks.q_network.apply(q_params, new_critic_obs, action)
        else:
            new_critic_obs = jnp.concatenate([new_state, new_critic_goal], axis=1)
            # hcb.id_print(new_critic_obs, what="\nnew_critic_obs")
            q_action = networks.q_network.apply(q_params, new_critic_obs, action)

        if len(q_action.shape) == 3:  # twin q trick
          assert q_action.shape[2] == 2
          q_action = jnp.min(q_action, axis=-1)

          # hcb.id_print(jnp.diag(q_action), what="\njnp.diag(q_action)")

        if config.exp_q_action:
            # hcb.id_print(q_action, what="(before) q_action")
            q_action = jnp.exp(q_action)
            # hcb.id_print(q_action, what="(after) q_action")

        if config.use_td:
            q_action = jnp.min(q_action, axis=-1)

            if config.sigmoid_q:
                q_action = jax.nn.sigmoid(q_action)

            actor_loss = alpha * log_prob - q_action
        else:
            actor_loss = alpha * log_prob - jnp.diag(q_action)

        if config.invert_actor_loss:
            # hcb.id_print(actor_loss, what="(before) actor_loss")
            actor_loss = -actor_loss
            # hcb.id_print(actor_loss, what="(after) actor_loss")

        # assert 0.0 <= config.bc_coef <= 1.0
        assert 0.0 <= config.bc_coef
        if config.bc_coef > 0:
          orig_action = transitions.action
          if config.random_goals == 0.5:
            orig_action = jnp.concatenate([orig_action, orig_action], axis=0)

          bc_loss = -1.0 * networks.log_prob(dist_params, orig_action)
          metrics = {"bc_loss": jnp.mean(bc_loss),
                     "actor_loss_nobc": jnp.mean(actor_loss)}
          actor_loss = (config.bc_coef * bc_loss
                        + (1 - config.bc_coef) * actor_loss)

      return jnp.mean(actor_loss), metrics

    alpha_grad = jax.value_and_grad(alpha_loss)
    critic_grad = jax.value_and_grad(critic_loss, has_aux=True)
    # actor_grad = jax.value_and_grad(actor_loss)
    actor_grad = jax.value_and_grad(actor_loss, has_aux=True)

    def update_step(
        state,
        all_transitions,
    ):
      transitions, val_transitions = all_transitions
      # transitions = all_transitions
      key, key_alpha, key_critic, key_actor = jax.random.split(state.key, 4)

      if adaptive_entropy_coefficient:
        alpha_loss, alpha_grads = alpha_grad(state.alpha_params,
                                             state.policy_params, transitions,
                                             key_alpha)
        alpha = jnp.exp(state.alpha_params)
      else:
        alpha = config.entropy_coefficient

      if not config.use_gcbc:
        (critic_loss, critic_metrics), critic_grads = critic_grad(
            state.q_params, state.policy_params, state.target_q_params,
            transitions, key_critic)

      # actor_loss, actor_grads = actor_grad(state.policy_params, state.q_params,
      #                                      alpha, transitions, key_actor, expert_goals)
      (actor_loss, actor_metrics), actor_grads = actor_grad(state.policy_params, state.q_params,
                                           alpha, transitions, key_actor, expert_goals)

      # Apply policy gradients
      actor_update, policy_optimizer_state = policy_optimizer.update(
          actor_grads, state.policy_optimizer_state)
      policy_params = optax.apply_updates(state.policy_params, actor_update)

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
        metrics.update(actor_metrics)

      # def val_metrics():
      #     (val_critic_loss, val_critic_metrics), _ = critic_grad(
      #         state.q_params, state.policy_params, state.target_q_params,
      #         val_transitions, key_critic)
      #
      #     val_actor_loss, _ = actor_grad(state.policy_params, state.q_params,
      #         alpha, val_transitions, key_actor, expert_goals)
      #
      #     return val_critic_loss, val_critic_metrics, val_actor_loss
      #
      # self.val_critic_loss = critic_loss
      # self.val_critic_metrics = critic_metrics
      # self.val_actor_loss = actor_loss
      #
      # val_critic_loss, val_critic_metrics, val_actor_loss = jax.lax.cond(
      #     state.steps % config.val_interval == 0,
      #     lambda _: val_metrics(),
      #     lambda _: (self.val_critic_loss, self.val_critic_metrics, self.val_actor_loss),
      #     operand=None)
      #
      # self.val_critic_loss = val_critic_loss
      # self.val_critic_metrics = val_critic_metrics
      # self.val_actor_loss = val_actor_loss
      metrics.update({
          'critic_loss': critic_loss,
          'actor_loss': actor_loss,
      })

      (val_critic_loss, val_critic_metrics), _ = critic_grad(
          state.q_params, state.policy_params, state.target_q_params,
          val_transitions, key_critic)

      (val_actor_loss, val_actor_metrics), _ = actor_grad(state.policy_params, state.q_params,
          alpha, val_transitions, key_actor, expert_goals)

      metrics.update({
          'val_critic_loss': val_critic_loss,
          'val_actor_loss': val_actor_loss,
      })
      metrics.update({"val_" + key:val for key, val in val_critic_metrics.items()})
      metrics.update({"val_" + key:val for key, val in val_actor_metrics.items()})

      new_state = TrainingState(
          policy_optimizer_state=policy_optimizer_state,
          q_optimizer_state=q_optimizer_state,
          policy_params=policy_params,
          q_params=q_params,
          target_q_params=new_target_q_params,
          key=key,
          steps=state.steps + 1,
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

      state = TrainingState(
          policy_optimizer_state=policy_optimizer_state,
          q_optimizer_state=q_optimizer_state,
          policy_params=policy_params,
          q_params=q_params,
          target_q_params=q_params,
          key=key,
          steps=0)

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
      # self._state, metrics = self._update_step(self._state, transitions)

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
