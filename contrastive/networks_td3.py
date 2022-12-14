"""TD3 networks definition."""
import dataclasses
from typing import Optional, Tuple, Callable, Sequence

from acme import specs
from acme import types
from acme.agents.jax import actor_core as actor_core_lib
from acme.jax import networks as networks_lib
from acme.jax import utils
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


@dataclasses.dataclass
class TD3Networks:
  """Network and pure functions for the TD3 agent."""
  policy_network: networks_lib.FeedForwardNetwork
  critic_network: networks_lib.FeedForwardNetwork
  twin_critic_network: networks_lib.FeedForwardNetwork
  r_network: networks_lib.FeedForwardNetwork
  add_policy_noise: Callable[[types.NestedArray, networks_lib.PRNGKey,
                              float, float], types.NestedArray]
  sample: networks_lib.SampleFn ###@@@###
  sample_eval: Optional[networks_lib.SampleFn] = None ###@@@###


def apply_policy_and_sample( ###@@@###
    networks,
    eval_mode = False):
  """Returns a function that computes actions."""
  sample_fn = networks.sample if not eval_mode else networks.sample_eval
  if not sample_fn:
    raise ValueError('sample function is not provided')

  def apply_and_sample(params, key, obs):
    return sample_fn(networks.policy_network.apply(params, obs), key)
  return apply_and_sample


def get_default_behavior_policy(
    networks: TD3Networks, action_specs: specs.BoundedArray,
    sigma: float) -> actor_core_lib.FeedForwardPolicy:
  """Selects action according to the policy."""
  def behavior_policy(params: networks_lib.Params, key: networks_lib.PRNGKey,
                      observation: types.NestedArray):
    action = networks.policy_network.apply(params, observation)
    noise = jax.random.normal(key, shape=action.shape) * sigma
    noisy_action = jnp.clip(action + noise,
                            action_specs.minimum, action_specs.maximum)
    return noisy_action
  return behavior_policy


def make_networks(
    spec: specs.EnvironmentSpec,
    obs_dim,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    use_image_obs = False,
    slice_actor_goal=False) -> TD3Networks:
  """Creates networks used by the agent.
  The networks used are based on LayerNormMLP, which is different than the
  MLP with relu activation described in TD3 (which empirically performs worse).
  Args:
    spec: Environment specs
    hidden_layer_sizes: list of sizes of hidden layers in actor/critic networks
  Returns:
    network: TD3Networks
  """

  action_specs = spec.actions
  num_dimensions = np.prod(action_specs.shape, dtype=int)

  TORSO = networks_lib.AtariTorso  # pylint: disable=invalid-name

  def _unflatten_obs(obs):
    # print("obs.shape:", obs.shape)

    if len(obs.shape) == 2:
        state = jnp.reshape(obs[:, :obs_dim], (-1, 64, 64, 3)) / 255.0
        goal = jnp.reshape(obs[:, obs_dim:], (-1, 64, 64, 3)) / 255.0
    elif len(obs.shape) == 1:
        state = jnp.reshape(obs[:obs_dim], (64, 64, 3)) / 255.0
        goal = jnp.reshape(obs[obs_dim:], (64, 64, 3)) / 255.0
    else:
        raise ValueError(f"obs.shape: {obs.shape}")

    return state, goal

  def add_policy_noise(action: types.NestedArray,
                       key: networks_lib.PRNGKey,
                       target_sigma: float,
                       noise_clip: float) -> types.NestedArray:
    """Adds action noise to bootstrapped Q-value estimate in critic loss."""
    noise = jax.random.normal(key=key, shape=action_specs.shape) * target_sigma
    noise = jnp.clip(noise, -noise_clip, noise_clip)
    return jnp.clip(action + noise, action_specs.minimum, action_specs.maximum)

  def _actor_fn(obs: types.NestedArray) -> types.NestedArray:
    # if use_image_obs:
    #   state, goal = _unflatten_obs(obs)
    #   # obs = jnp.concatenate([state, goal], axis=-1)
    #   obs = state
    #   obs = TORSO()(obs)

    # obs = jnp.concatenate([obs[:, :obs_dim], np.zeros_like(obs[:, :obs_dim])], axis=-1)
    if use_image_obs:
      state, goal = _unflatten_obs(obs)
      # obs = jnp.concatenate([state, goal], axis=-1)
      # obs = state

      if slice_actor_goal:
          obs = state
      else:
          obs = jnp.concatenate([state, goal], axis=-1)

      obs = TORSO()(obs)

    network = hk.Sequential([
        networks_lib.LayerNormMLP(hidden_layer_sizes,
                                  activate_final=True),
        networks_lib.NearZeroInitializedLinear(num_dimensions),
        networks_lib.TanhToSpec(spec.actions),
    ])
    return network(obs)


  def _critic_fn(obs: types.NestedArray,
                 action: types.NestedArray) -> types.NestedArray:


    if use_image_obs:
      state, goal = _unflatten_obs(obs)
      # obs = jnp.concatenate([state, goal], axis=-1)
      obs = state
      obs = TORSO()(obs)

    network1 = hk.Sequential([
        networks_lib.LayerNormMLP(list(hidden_layer_sizes) + [1]),
    ])
    input_ = jnp.concatenate([obs, action], axis=-1)
    value = network1(input_)
    return jnp.squeeze(value)

  def _reward_fn(obs):
    if use_image_obs:
      state, goal = _unflatten_obs(obs)
      # obs = jnp.concatenate([state, goal], axis=-1)
      obs = state
      obs = TORSO()(obs)
    network = hk.Sequential([
        hk.nets.MLP(
            list(hidden_layer_sizes) + [1],
            w_init=hk.initializers.VarianceScaling(1.0, 'fan_in', 'uniform'),
            activation=jax.nn.relu,
            # activate_final=True),
            activate_final=False),
        # networks_lib.NormalTanhDistribution(num_dimensions,
        #                                     min_scale=actor_min_std),
    ])
    return network(obs)


  policy = hk.without_apply_rng(hk.transform(_actor_fn))
  critic = hk.without_apply_rng(hk.transform(_critic_fn))
  reward = hk.without_apply_rng(hk.transform(_reward_fn))

  # Create dummy observations and actions to create network parameters.
  dummy_action = utils.zeros_like(spec.actions)
  dummy_obs = utils.zeros_like(spec.observations)
  dummy_action = utils.add_batch_dim(dummy_action)
  dummy_obs = utils.add_batch_dim(dummy_obs)

  actor_dummy_obs = dummy_obs[:, :obs_dim] if slice_actor_goal else dummy_obs

  network = TD3Networks(
      policy_network=networks_lib.FeedForwardNetwork(
          lambda key: policy.init(key, actor_dummy_obs), policy.apply),
      critic_network=networks_lib.FeedForwardNetwork(
          lambda key: critic.init(key, dummy_obs, dummy_action), critic.apply),
      twin_critic_network=networks_lib.FeedForwardNetwork(
          lambda key: critic.init(key, dummy_obs, dummy_action), critic.apply),
      r_network=networks_lib.FeedForwardNetwork(
          lambda key: reward.init(key, dummy_obs), reward.apply),
      add_policy_noise=add_policy_noise,
      sample=lambda params, key: params.sample(seed=key),
      # sample_eval=lambda params, key: params.mode(),
      sample_eval=lambda params, key: params,
      )

  return network
