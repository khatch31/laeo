from typing import Iterator, List, Optional

from acme import adders
from acme import core
from acme import specs
from acme.adders import reverb as adders_reverb
from acme.agents.jax import actor_core as actor_core_lib
from acme.agents.jax import actors
from acme.agents.jax import builders
from acme.agents.jax.td3 import config as td3_config
from acme.agents.jax.td3 import learning
from acme.agents.jax.td3 import networks as td3_networks
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

from acme.agents.jax import td3

class ContrastiveBuilderGoalsTD3(td3.builder.TD3Builder):
  """TD3 Builder."""

  def __init__(
      self,
      config,
      logger_fn = lambda: None,
      save_data=False,
      data_save_dir=None
   ):
      super(ContrastiveBuilderGoalsTD3, self).__init__(config)
      self._logger_fn = logger_fn
      self._save_data = save_data
      self._data_save_dir = data_save_dir



  # def make_learner(
  #     self,
  #     random_key: networks_lib.PRNGKey,
  #     networks: td3_networks.TD3Networks,
  #     dataset: Iterator[reverb.ReplaySample],
  #     logger_fn: loggers.LoggerFactory,
  #     environment_spec: specs.EnvironmentSpec,
  #     replay_client: Optional[reverb.Client] = None,
  #     counter: Optional[counting.Counter] = None,
  # ) -> core.Learner:
  #   del environment_spec, replay_client
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

    # return learning.TD3Learner(
    #     networks=networks,
    #     random_key=random_key,
    #     discount=self._config.discount,
    #     target_sigma=self._config.target_sigma,
    #     noise_clip=self._config.noise_clip,
    #     policy_optimizer=policy_optimizer,
    #     critic_optimizer=critic_optimizer,
    #     twin_critic_optimizer=twin_critic_optimizer,
    #     num_sgd_steps_per_step=self._config.num_sgd_steps_per_step,
    #     # bc_alpha=self._config.bc_alpha,
    #     iterator=dataset,
    #     # logger=logger_fn('learner'),
    #     logger=self._logger_fn(),
    #     counter=counter)
    return learning.TD3Learner(
        networks=networks,
        random_key=random_key,
        discount=self._config.discount,
        iterator=dataset,
        policy_optimizer=policy_optimizer,
        critic_optimizer=critic_optimizer,
        twin_critic_optimizer=twin_critic_optimizer,
        delay=self._config.delay,
        target_sigma=self._config.target_sigma,
        noise_clip=self._config.noise_clip,
        tau=self._config.tau,
        bc_alpha=self._config.bc_alpha,
        counter=counter,
        logger=self._logger_fn(),
        num_sgd_steps_per_step=self._config.num_sgd_steps_per_step,)
