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

"""Contrastive RL networks definition."""
import dataclasses
from typing import Optional, Tuple, Callable

from acme import specs
from acme.agents.jax import actor_core as actor_core_lib
from acme.jax import networks as networks_lib
from acme.jax import utils
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

import os
from glob import glob
import shutil

@dataclasses.dataclass
class ContrastiveNetworks:
  """Network and pure functions for the Contrastive RL agent."""
  policy_network: networks_lib.FeedForwardNetwork
  q_network: networks_lib.FeedForwardNetwork
  # r_network: networks_lib.FeedForwardNetwork
  log_prob: networks_lib.LogProbFn
  repr_fn: Callable[Ellipsis, networks_lib.NetworkOutput]
  sample: networks_lib.SampleFn
  sample_eval: Optional[networks_lib.SampleFn] = None


def apply_policy_and_sample(
    networks,
    eval_mode = False):
  """Returns a function that computes actions."""
  sample_fn = networks.sample if not eval_mode else networks.sample_eval
  if not sample_fn:
    raise ValueError('sample function is not provided')

  def apply_and_sample(params, key, obs):
    return sample_fn(networks.policy_network.apply(params, obs), key)
  return apply_and_sample

def make_networks(
    spec,
    obs_dim,
    repr_dim = 64,
    repr_norm = False,
    repr_norm_temp = True,
    hidden_layer_sizes = (256, 256),
    actor_min_std = 1e-6,
    twin_q = False,
    use_image_obs = False,
    use_td = False,
    slice_actor_goal=False):
  """Creates networks used by the agent."""

  print(f"repr_norm: {repr_norm}, repr_norm_temp: {repr_norm_temp}")

  num_dimensions = np.prod(spec.actions.shape, dtype=int)
  TORSO = networks_lib.AtariTorso  # pylint: disable=invalid-name

  def _unflatten_obs(obs):
    state = jnp.reshape(obs[:, :obs_dim], (-1, 64, 64, 3)) / 255.0
    goal = jnp.reshape(obs[:, obs_dim:], (-1, 64, 64, 3)) / 255.0
    return state, goal

  def _repr_fn(obs, action, hidden=None):
    # The optional input hidden is the image representations. We include this
    # as an input for the second Q value when twin_q = True, so that the two Q
    # values use the same underlying image representation.
    if hidden is None:
      if use_image_obs:
        state, goal = _unflatten_obs(obs)
        img_encoder = TORSO()
        state = img_encoder(state)
        goal = img_encoder(goal)
      else:
        state = obs[:, :obs_dim]
        goal = obs[:, obs_dim:]
    else:
      state, goal = hidden

    sa_encoder = hk.nets.MLP(
        list(hidden_layer_sizes) + [repr_dim],
        w_init=hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform'),
        activation=jax.nn.relu,
        name='sa_encoder')
    sa_repr = sa_encoder(jnp.concatenate([state, action], axis=-1))

    g_encoder = hk.nets.MLP(
        list(hidden_layer_sizes) + [repr_dim],
        w_init=hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform'),
        activation=jax.nn.relu,
        name='g_encoder')
    g_repr = g_encoder(goal)

    if repr_norm:
      sa_repr = sa_repr / jnp.linalg.norm(sa_repr, axis=1, keepdims=True)
      g_repr = g_repr / jnp.linalg.norm(g_repr, axis=1, keepdims=True)

      if repr_norm_temp:
        log_scale = hk.get_parameter('repr_log_scale', [], dtype=sa_repr.dtype,
                                     init=jnp.zeros)
        sa_repr = sa_repr / jnp.exp(log_scale)
    return sa_repr, g_repr, (state, goal)

  def _combine_repr(sa_repr, g_repr):
    return jax.numpy.einsum('ik,jk->ij', sa_repr, g_repr)

  def _critic_fn(obs, action):
    # print(f"[_critic_fn] obs: {obs}, obs.shape: {obs.shape}")
    # print(f"[_critic_fn] action: {action}, action.shape: {action.shape}")
    sa_repr, g_repr, hidden = _repr_fn(obs, action)
    outer = _combine_repr(sa_repr, g_repr)
    if twin_q:
      # print(f"[_critic_fn] obs: {obs}, obs.shape: {obs.shape}")
      # print(f"[_critic_fn] action: {action}, action.shape: {action.shape}")
      sa_repr2, g_repr2, _ = _repr_fn(obs, action, hidden=hidden)
      outer2 = _combine_repr(sa_repr2, g_repr2)
      # outer.shape = [batch_size, batch_size, 2]
      outer = jnp.stack([outer, outer2], axis=-1)
    return outer

  def _critic_fn_td_single(obs, action):
    if use_image_obs:
      state, goal = _unflatten_obs(obs)
      obs = state
      obs = TORSO()(obs)
    else:
      obs = obs[:, :obs_dim]

    network = hk.Sequential([
        hk.nets.MLP(
            list(hidden_layer_sizes) + [1],
            w_init=hk.initializers.VarianceScaling(1.0, 'fan_in', 'uniform'),
            activation=jax.nn.relu,
            activate_final=False),
    ])
    return network(jnp.concatenate([obs, action], axis=-1))

  def _critic_fn_td(obs, action):
      out = _critic_fn_td_single(obs, action)
      if twin_q:
          out2 = _critic_fn_td_single(obs, action)
          out = jnp.concatenate([out, out2], axis=-1)
      return out

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

  def _actor_fn(obs):
    if use_image_obs:
      state, goal = _unflatten_obs(obs)

      if slice_actor_goal:
          obs = state
      else:
          obs = jnp.concatenate([state, goal], axis=-1)

      # # if obs.shape[0] == 1:
      # #     save_dir = os.path.join(os.getcwd(), "debug_images", "eval")
      # #     existing_files = glob(os.path.join(save_dir, "**", "img_*.npy"), recursive=True)
      # #     obs_numpy = np.squeeze(np.asarray(obs))
      # #     im_no = len(existing_files)
      # #     if im_no < 50:
      # #         os.makedirs(save_dir, exist_ok=True)
      # #         np.save(os.path.join(save_dir, f"img_{im_no}"), obs_numpy)
      # #         print(f"Saved image to \"{os.path.join(save_dir, f'img_{im_no}.npy')}\".")
      # #     else:
      # #         print(f"Removing \"{save_dir}\"...")
      # #         shutil.rmtree(save_dir)
      # # else:
      # #     batch_size = obs.shape[0]
      # #     save_dir = os.path.join(os.getcwd(), "debug_images", "train")
      # #     existing_files = glob(os.path.join(save_dir, "**", "img_*.npy"), recursive=True)
      # #     obs_numpy = np.asarray(obs)
      # #     im_no = len(existing_files)
      # #     if im_no < batch_size:
      # #         os.makedirs(save_dir, exist_ok=True)
      # #         for i in range(batch_size):
      # #             np.save(os.path.join(save_dir, f"img_{im_no}"), obs_numpy[0])
      # #             print(f"Saved image to \"{os.path.join(save_dir, f'img_{i}.npy')}\".")
      # #     else:
      # #         print(f"Removing \"{save_dir}\"...")
      # #         shutil.rmtree(save_dir)
      # if obs.shape[0] == 1:
      #     save_dir = os.path.join(os.getcwd(), "debug_images", "eval")
      #     existing_files = glob(os.path.join(save_dir, "**", "img_*.npy"), recursive=True)
      #     obs_numpy = np.squeeze(np.asarray(obs))
      #     im_no = len(existing_files)
      #     os.makedirs(save_dir, exist_ok=True)
      #     np.save(os.path.join(save_dir, f"img_{im_no}"), obs_numpy)
      #     print(f"Saved image to \"{os.path.join(save_dir, f'img_{im_no}.npy')}\".")
      # else:
      #     batch_size = obs.shape[0]
      #     save_dir = os.path.join(os.getcwd(), "debug_images", "train")
      #     existing_files = glob(os.path.join(save_dir, "**", "batch_*.npy"), recursive=True)
      #     obs_numpy = np.asarray(obs)
      #     batch_no = len(existing_files)
      #     os.makedirs(save_dir, exist_ok=True)
      #     np.save(os.path.join(save_dir, f"batch_{batch_no}"), obs_numpy)
      #     print(f"Saved batch to \"{os.path.join(save_dir, f'batch_{batch_no}.npy')}\".")


      obs = TORSO()(obs)

    network = hk.Sequential([
        hk.nets.MLP(
            list(hidden_layer_sizes),
            w_init=hk.initializers.VarianceScaling(1.0, 'fan_in', 'uniform'),
            activation=jax.nn.relu,
            activate_final=True),
        networks_lib.NormalTanhDistribution(num_dimensions,
                                            min_scale=actor_min_std),
    ])
    # print(f"[_actor_fn] obs: {obs}, obs.shape: {obs.shape}")
    return network(obs)

  policy = hk.without_apply_rng(hk.transform(_actor_fn))
  critic = hk.without_apply_rng(hk.transform(_critic_fn_td if use_td else _critic_fn))
  reward = hk.without_apply_rng(hk.transform(_reward_fn))
  repr_fn = hk.without_apply_rng(hk.transform(_repr_fn))

  # Create dummy observations and actions to create network parameters.
  dummy_action = utils.zeros_like(spec.actions)
  dummy_obs = utils.zeros_like(spec.observations)#.astype(np.float32)
  dummy_action = utils.add_batch_dim(dummy_action)
  dummy_obs = utils.add_batch_dim(dummy_obs)
  actor_dummy_obs = dummy_obs[:, :obs_dim] if slice_actor_goal else dummy_obs

  return ContrastiveNetworks(
      # policy_network=networks_lib.FeedForwardNetwork(
      #     lambda key: policy.init(key, dummy_obs), policy.apply),
      policy_network=networks_lib.FeedForwardNetwork(
          lambda key: policy.init(key, actor_dummy_obs), policy.apply),
      q_network=networks_lib.FeedForwardNetwork(
          lambda key: critic.init(key, dummy_obs, dummy_action), critic.apply),
      # # r_network=networks_lib.FeedForwardNetwork(
      # #     lambda key: reward.init(key, dummy_obs, dummy_action), reward.apply),
      # r_network=networks_lib.FeedForwardNetwork(
      #     lambda key: reward.init(key, dummy_obs[:, :obs_dim]), reward.apply),
      repr_fn=repr_fn.apply,
      log_prob=lambda params, actions: params.log_prob(actions),
      sample=lambda params, key: params.sample(seed=key),
      sample_eval=lambda params, key: params.mode(),
      )
