"""TD3 learner implementation."""

import time
from typing import Dict, Iterator, List, NamedTuple, Optional, Tuple

import acme
from acme import types
# from acme.agents.jax.td3 import networks as td3_networks
from contrastive import networks_td3 as td3_networks ###@@@###
from acme.jax import networks as networks_lib
from acme.jax import utils
from acme.utils import counting
from acme.utils import loggers
import jax
import jax.numpy as jnp
import optax
import reverb
import rlax

from jax.experimental import host_callback as hcb
from contrastive.default_logger import make_default_logger

from contrastive.losses import sigmoid_positive_unlabeled_loss

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

  r_optimizer_state: optax.OptState
  r_params: networks_lib.Params

  steps: int
  random_key: networks_lib.PRNGKey


class TD3Learner(acme.Learner):
  """TD3 learner."""

  _state: TrainingState

  def __init__(self,
               networks: td3_networks.TD3Networks,
               random_key: networks_lib.PRNGKey,
               discount: float,
               iterator: Iterator[reverb.ReplaySample],
               val_iterator: Iterator[reverb.ReplaySample],
               policy_optimizer: optax.GradientTransformation,
               critic_optimizer: optax.GradientTransformation,
               twin_critic_optimizer: optax.GradientTransformation,
               r_optimizer: optax.GradientTransformation,
               delay: int = 2,
               target_sigma: float = 0.2,
               noise_clip: float = 0.5,
               tau: float = 0.005,
               use_sarsa_target: bool = False,
               bc_alpha: Optional[float] = None,
               config=None,
               counter: Optional[counting.Counter] = None,
               logger: Optional[loggers.Logger] = None,
               num_sgd_steps_per_step: int = 1,
               expert_goals=None):
    """Initializes the TD3 learner.
    Args:
      networks: TD3 networks.
      random_key: a key for random number generation.
      discount: discount to use for TD updates
      iterator: an iterator over training data.
      policy_optimizer: the policy optimizer.
      critic_optimizer: the Q-function optimizer.
      twin_critic_optimizer: the twin Q-function optimizer.
      delay: ratio of policy updates for critic updates (see TD3),
        delay=2 means 2 updates of the critic for 1 policy update.
      target_sigma: std of zero mean Gaussian added to the action of
        the next_state, for critic evaluation (reducing overestimation bias).
      noise_clip: hard constraint on target noise.
      tau: target parameters smoothing coefficient.
      use_sarsa_target: compute on-policy target using iterator's actions rather
        than sampled actions.
        Useful for 1-step offline RL (https://arxiv.org/pdf/2106.08909.pdf).
        When set to `True`, `target_policy_params` are unused.
        This is only working when the learner is used as an offline algorithm.
        I.e. TD3Builder does not support adding the SARSA target to the replay
        buffer.
      bc_alpha: bc_alpha: Implements TD3+BC.
        See comments in TD3Config.bc_alpha for details.
      counter: counter object used to keep track of steps.
      logger: logger object to be used by learner.
      num_sgd_steps_per_step: number of sgd steps to perform per learner 'step'.
    """

    expert_goals = jnp.array(expert_goals, dtype=jnp.float32)

    def policy_loss(
        policy_params: networks_lib.Params,
        critic_params: networks_lib.Params,
        transition: types.NestedArray,
    ) -> jnp.ndarray:
      # Computes the discrete policy gradient loss.

      # action = networks.policy_network.apply(policy_params, transition.observation)
      new_obs = jnp.concatenate((transition.observation[:, :config.obs_dim], jnp.zeros_like(transition.observation[:, config.obs_dim:])), axis=-1)
      action = networks.policy_network.apply(policy_params, new_obs)
      grad_critic = jax.vmap(
          jax.grad(networks.critic_network.apply, argnums=2),
          in_axes=(None, 0, 0))
      # dq_da = grad_critic(critic_params, transition.observation, action)
      dq_da = grad_critic(critic_params, new_obs, action)
      batch_dpg_learning = jax.vmap(rlax.dpg_loss, in_axes=(0, 0))
      loss = jnp.mean(batch_dpg_learning(action, dq_da))
      if bc_alpha is not None:
        # BC regularization for offline RL
        # q_sa = networks.critic_network.apply(critic_params, transition.observation, action)
        q_sa = networks.critic_network.apply(critic_params, new_obs, action)
        bc_factor = jax.lax.stop_gradient(bc_alpha / jnp.mean(jnp.abs(q_sa)))
        loss += jnp.mean(jnp.square(action - transition.action)) / bc_factor
      return loss

    def critic_loss(
        critic_params: networks_lib.Params,
        r_params: networks_lib.Params,
        state: TrainingState,
        transition: types.Transition,
        random_key: jnp.ndarray,
    ):
      # Computes the critic loss.
      print("transition.observation.shape:", transition.observation.shape)
      # q_tm1 = networks.critic_network.apply(critic_params, transition.observation, transition.action)
      new_obs = jnp.concatenate((transition.observation[:, :config.obs_dim], jnp.zeros_like(transition.observation[:, config.obs_dim:])), axis=-1)
      q_tm1 = networks.critic_network.apply(critic_params, new_obs, transition.action)

      if use_sarsa_target:
        # TODO(b/222674779): use N-steps Trajectories to get the next actions.
        assert 'next_action' in transition.extras, (
            'next actions should be given as extras for one step RL.')
        action = transition.extras['next_action']
      else:
        # action = networks.policy_network.apply(state.target_policy_params, transition.next_observation)
        new_next_obs = jnp.concatenate((transition.next_observation[:, :config.obs_dim], jnp.zeros_like(transition.next_observation[:, config.obs_dim:])), axis=-1)
        action = networks.policy_network.apply(state.target_policy_params, new_next_obs)
        action = networks.add_policy_noise(action, random_key,
                                           target_sigma, noise_clip)

      new_next_obs = jnp.concatenate((transition.next_observation[:, :config.obs_dim], jnp.zeros_like(transition.next_observation[:, config.obs_dim:])), axis=-1)
      q_t = networks.critic_network.apply(
          state.target_critic_params,
          # transition.next_observation,
          new_next_obs,
          action)
      twin_q_t = networks.twin_critic_network.apply(
          state.target_twin_critic_params,
          # transition.next_observation,
          new_next_obs,
          action)

      q_t = jnp.minimum(q_t, twin_q_t)

      reward_logits = networks.r_network.apply(r_params, new_obs)[:, 0]
      predicted_reward = jax.nn.sigmoid(reward_logits)
      predicted_reward = jax.lax.stop_gradient(predicted_reward)

      if config.use_true_reward:
          reward = transition.reward
      elif config.use_l2_reward:
          l2goal = jnp.ones_like(transition.observation[:, :3]) * jnp.array([1.3, 0.3, 0.9])
          reward = -jnp.linalg.norm(transition.observation[:, :3] - l2goal, axis=-1)
      else:
          reward = predicted_reward

          if config.shift_learned_reward:
              reward -= 0.5
              reward *= 2

      # print("reward:", reward)
      # hcb.id_print(reward, what="reward")

      target_q_tm1 = transition.reward + discount * transition.discount * q_t
      td_error = jax.lax.stop_gradient(target_q_tm1) - q_tm1

      metrics = {
        "logits":q_t.mean(),
        "sigmoid_logits":jax.nn.sigmoid(q_t).mean(),

        "predicted_reward_logits":reward_logits.mean(),
        "predicted_reward":predicted_reward.mean(),
        'predicted_reward_binary_accuracy': jnp.mean((reward_logits > 0) == transition.reward),
        'predicted_reward_sigmoid_binary_accuracy': jnp.mean((predicted_reward > 0.5) == transition.reward),
      }

      return jnp.mean(jnp.square(td_error)), metrics

    def reward_loss(r_params,
                    transitions,
                    key,
                    expert_goals):
      batch_size = transitions.observation.shape[0]

      # Negative = policy, positive = expert goals

      s, g = jnp.split(transitions.observation, [config.obs_dim], axis=1)

      idxs = jax.random.randint(key, (batch_size,), 0, expert_goals.shape[0])
      obs_and_goals = jnp.concatenate([s, expert_goals[idxs]], axis=0)
      new_obs_and_goals = jnp.concatenate((obs_and_goals, jnp.zeros_like(obs_and_goals)), axis=-1)
      logits = networks.r_network.apply(r_params, new_obs_and_goals)[:, 0]
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

    def update_step(
        state: TrainingState,
        all_transitions,
    ) -> Tuple[TrainingState, Dict[str, jnp.ndarray]]:

      transitions, val_transitions = all_transitions
      # transitions = all_transitions

      random_key, key_critic, key_twin, key_reward = jax.random.split(state.random_key, 4)

      # Updates on the critic: compute the gradients, and update using
      # Polyak averaging.
      critic_loss_and_grad = jax.value_and_grad(critic_loss, has_aux=True)
      # critic_loss_value, critic_gradients = critic_loss_and_grad(
      #     state.critic_params, state, transitions, key_critic)
      (critic_loss_value, critic_metrics), critic_gradients = critic_loss_and_grad(
          state.critic_params, state.r_params, state, transitions, key_critic)
      critic_updates, critic_opt_state = critic_optimizer.update(
          critic_gradients, state.critic_opt_state)
      critic_params = optax.apply_updates(state.critic_params, critic_updates)
      # In the original authors' implementation the critic target update is
      # delayed similarly to the policy update which we found empirically to
      # perform slightly worse.
      target_critic_params = optax.incremental_update(
          new_tensors=critic_params,
          old_tensors=state.target_critic_params,
          step_size=tau)

      # Updates on the twin critic: compute the gradients, and update using
      # Polyak averaging.
      (twin_critic_loss_value, twin_critic_metrics), twin_critic_gradients = critic_loss_and_grad(
          state.twin_critic_params, state.r_params, state, transitions, key_twin)
      # twin_critic_loss_value, twin_critic_gradients = critic_loss_and_grad(
      #     state.twin_critic_params, state, transitions, key_twin)
      twin_critic_updates, twin_critic_opt_state = twin_critic_optimizer.update(
          twin_critic_gradients, state.twin_critic_opt_state)
      twin_critic_params = optax.apply_updates(state.twin_critic_params,
                                               twin_critic_updates)
      # In the original authors' implementation the twin critic target update is
      # delayed similarly to the policy update which we found empirically to
      # perform slightly worse.
      target_twin_critic_params = optax.incremental_update(
          new_tensors=twin_critic_params,
          old_tensors=state.target_twin_critic_params,
          step_size=tau)

      # Updates on the policy: compute the gradients, and update using
      # Polyak averaging (if delay enabled, the update might not be applied).
      policy_loss_and_grad = jax.value_and_grad(policy_loss)
      policy_loss_value, policy_gradients = policy_loss_and_grad(
          state.policy_params, state.critic_params, transitions)
      def update_policy_step():
        policy_updates, policy_opt_state = policy_optimizer.update(
            policy_gradients, state.policy_opt_state)
        policy_params = optax.apply_updates(state.policy_params, policy_updates)
        target_policy_params = optax.incremental_update(
            new_tensors=policy_params,
            old_tensors=state.target_policy_params,
            step_size=tau)
        return policy_params, target_policy_params, policy_opt_state

      # The update on the policy is applied every `delay` steps.
      current_policy_state = (state.policy_params, state.target_policy_params,
                              state.policy_opt_state)
      policy_params, target_policy_params, policy_opt_state = jax.lax.cond(
          state.steps % delay == 0,
          lambda _: update_policy_step(),
          lambda _: current_policy_state,
          operand=None)

      steps = state.steps + 1


      reward_grad = jax.value_and_grad(reward_loss, has_aux=True)

      (val_reward_loss, val_reward_metrics), _ = reward_grad(
          state.r_params, val_transitions, key_reward, expert_goals)

      (reward_loss_value, reward_metrics), reward_grads = reward_grad(
          state.r_params, transitions, key_reward, expert_goals)

      # Apply reward gradients
      reward_update, r_optimizer_state = r_optimizer.update(
          reward_grads, state.r_optimizer_state)
      r_params = optax.apply_updates(state.r_params, reward_update)

      new_state = TrainingState(
          policy_params=policy_params,
          critic_params=critic_params,
          twin_critic_params=twin_critic_params,
          target_policy_params=target_policy_params,
          target_critic_params=target_critic_params,
          target_twin_critic_params=target_twin_critic_params,
          policy_opt_state=policy_opt_state,
          critic_opt_state=critic_opt_state,
          twin_critic_opt_state=twin_critic_opt_state,

          r_optimizer_state=r_optimizer_state,
          r_params=r_params,

          steps=steps,
          random_key=random_key,
      )

      metrics = {
          'policy_loss': policy_loss_value,
          'critic_loss': critic_loss_value,
          'twin_critic_loss': twin_critic_loss_value,
          'reward_loss': reward_loss_value,
      }

      metrics.update({"train_" + key:val for key, val in reward_metrics.items()})
      metrics.update({"val_" + key:val for key, val in val_reward_metrics.items()})
      metrics.update(critic_metrics)
      metrics.update({"twin_" + key:val for key, val in twin_critic_metrics.items()})

      return new_state, metrics

    # General learner book-keeping and loggers.
    self._counter = counter or counting.Counter()
    # self._logger = logger or make_default_logger(
    #     'learner',
    #     asynchronous=True,
    #     serialize_fn=utils.fetch_devicearray,
    #     steps_key=self._counter.get_steps_key())
    self._logger = logger or make_default_logger(
        "~/acme",
        'learner', asynchronous=True, serialize_fn=utils.fetch_devicearray,
        time_delta=10.0,
        wandblogger=None)

    # Create prefetching dataset iterator.
    self._iterator = iterator
    self._val_iterator = val_iterator

    # Faster sgd step
    update_step = utils.process_multiple_batches(update_step,
                                                 num_sgd_steps_per_step)
    # Use the JIT compiler.
    self._update_step = jax.jit(update_step)

    key_init_policy, key_init_twin, key_init_target, key_state, key_r = jax.random.split(random_key, 5)
    # Create the network parameters and copy into the target network parameters.
    initial_policy_params = networks.policy_network.init(key_init_policy)
    initial_critic_params = networks.critic_network.init(key_init_twin)
    initial_twin_critic_params = networks.twin_critic_network.init(
        key_init_target)

    initial_target_policy_params = initial_policy_params
    initial_target_critic_params = initial_critic_params
    initial_target_twin_critic_params = initial_twin_critic_params

    # Initialize optimizers.
    initial_policy_opt_state = policy_optimizer.init(initial_policy_params)
    initial_critic_opt_state = critic_optimizer.init(initial_critic_params)
    initial_twin_critic_opt_state = twin_critic_optimizer.init(
        initial_twin_critic_params)

    initial_r_params = networks.r_network.init(key_r)
    initial_r_optimizer_state = r_optimizer.init(initial_r_params)

    # Create initial state.
    self._state = TrainingState(
        policy_params=initial_policy_params,
        target_policy_params=initial_target_policy_params,
        critic_params=initial_critic_params,
        twin_critic_params=initial_twin_critic_params,
        target_critic_params=initial_target_critic_params,
        target_twin_critic_params=initial_target_twin_critic_params,
        policy_opt_state=initial_policy_opt_state,
        critic_opt_state=initial_critic_opt_state,
        twin_critic_opt_state=initial_twin_critic_opt_state,

        r_params=initial_r_params,
        r_optimizer_state=initial_r_optimizer_state,

        steps=0,
        random_key=key_state
    )

    # Do not record timestamps until after the first learning step is done.
    # This is to avoid including the time it takes for actors to come online and
    # fill the replay buffer.
    self._timestamp = None

  def step(self):
    # Get data from replay (dropping extras if any). Note there is no
    # extra data here because we do not insert any into Reverb.
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

    # Attempts to write the logs.
    self._logger.write({**metrics, **counts})

  def get_variables(self, names: List[str]) -> List[networks_lib.Params]:
    variables = {
        'policy': self._state.policy_params,
        'critic': self._state.critic_params,
        'twin_critic': self._state.twin_critic_params,
    }
    return [variables[name] for name in names]

  def save(self) -> TrainingState:
    return self._state

  def restore(self, state: TrainingState):
    self._state = state
