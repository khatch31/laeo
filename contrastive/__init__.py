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

"""Contrastive RL agent."""

from contrastive.agents import DistributedContrastive
from contrastive.builder import ContrastiveBuilder
from contrastive.config import ContrastiveConfig
from contrastive.config_goals import ContrastiveConfigGoals
from contrastive.config_reward import ContrastiveConfigReward
from contrastive.config_goals_frozen_critic import ContrastiveConfigGoalsFrozenCritic
from contrastive.config import target_entropy_from_env_spec
from contrastive.learning import ContrastiveLearner
from contrastive.networks import apply_policy_and_sample
from contrastive.networks import ContrastiveNetworks
from contrastive.networks import make_networks


###===###
from contrastive.learning_goals import ContrastiveLearnerGoals
from contrastive.agents_goals import DistributedContrastiveGoals
from contrastive.builder_goals import ContrastiveBuilderGoals

from contrastive.learning_reward import ContrastiveLearnerReward
from contrastive.agents_reward import DistributedContrastiveReward
from contrastive.builder_reward import ContrastiveBuilderReward

from contrastive.learning_goals_frozen_critic import ContrastiveLearnerGoalsFrozenCritic
from contrastive.agents_goals_frozen_critic import DistributedContrastiveGoalsFrozenCritic
from contrastive.builder_goals_frozen_critic import ContrastiveBuilderGoalsFrozenCritic

# from contrastive.losses import sigmoid_positive_unlabeled_loss

# from contrastive.episode_saver_adder import EpisodeAdderSaver
###---###
