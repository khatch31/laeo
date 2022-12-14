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

"""Utility for loading the goal-conditioned environments."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import ant_env
import fetch_envs
import gym
# import sys; sys.path.append("/iris/u/khatch/metaworld_door")
import metaworld
import numpy as np
import point_env
import point_env_fixed_goal

os.environ['SDL_VIDEODRIVER'] = 'dummy'

import ceborl_envs


def euler2quat(euler):
  """Convert Euler angles to quaternions."""
  euler = np.asarray(euler, dtype=np.float64)
  assert euler.shape[-1] == 3, 'Invalid shape euler {}'.format(euler)

  ai, aj, ak = euler[Ellipsis, 2] / 2, -euler[Ellipsis, 1] / 2, euler[Ellipsis, 0] / 2
  si, sj, sk = np.sin(ai), np.sin(aj), np.sin(ak)
  ci, cj, ck = np.cos(ai), np.cos(aj), np.cos(ak)
  cc, cs = ci * ck, ci * sk
  sc, ss = si * ck, si * sk

  quat = np.empty(euler.shape[:-1] + (4,), dtype=np.float64)
  quat[Ellipsis, 0] = cj * cc + sj * ss
  quat[Ellipsis, 3] = cj * sc - sj * cs
  quat[Ellipsis, 2] = -(cj * ss + sj * cc)
  quat[Ellipsis, 1] = cj * cs - sj * sc
  return quat


def load(env_name):
  """Loads the train and eval environments, as well as the obs_dim."""
  # pylint: disable=invalid-name
  kwargs = {}

  if "sawyer" in env_name:
      if "offline" in env_name:
          env_name = env_name[:][len("offline_"):]
          print("env_name:", env_name)

      if env_name == 'sawyer_push':
        CLASS = SawyerPush
        max_episode_steps = 150
      if env_name == 'sawyer_push-goals':
        CLASS = SawyerPushGoals
        max_episode_steps = 150
      elif env_name == 'sawyer_drawer':
        CLASS = SawyerDrawer
        max_episode_steps = 150
      elif env_name == "sawyer_drawer-goals":
        CLASS = SawyerDrawerGoals
        max_episode_steps = 150
      elif env_name == 'sawyer_drawer_image':
        CLASS = SawyerDrawerImage
        max_episode_steps = 50
        kwargs['task'] = 'openclose'
      elif env_name == 'sawyer_window_image':
        CLASS = SawyerWindowImage
        kwargs['task'] = 'openclose'
        max_episode_steps = 50
      elif env_name == 'sawyer_push_image':
        CLASS = SawyerPushImage
        max_episode_steps = 150
        kwargs['start_at_obj'] = True
      elif env_name == 'sawyer_bin':
        CLASS = SawyerBin
        max_episode_steps = 150
      elif env_name == 'sawyer_bin-goals':
        CLASS = SawyerBinGoals
        max_episode_steps = 150
      elif env_name == 'sawyer_bin_image':
        CLASS = SawyerBinImage
        max_episode_steps = 150
      elif env_name == 'sawyer_window':
        CLASS = SawyerWindow
        max_episode_steps = 150
      elif env_name == 'sawyer_window-goals':
        CLASS = SawyerWindowGoals
        max_episode_steps = 150
      elif env_name == "sawyer_window_image_minimal-goals":
        CLASS = SawyerWindowImageMinimalGoals
        max_episode_steps = 150
      # elif env_name == "sawyer_lever_pull-goals":
      #   CLASS = SawyerLeverPullGoalsV2
      #   max_episode_steps = 150
      elif "sawyer_lever_pull-goals" in env_name:
        CLASS = SawyerLeverPullGoalsV2
        max_episode_steps = 150

        if "no-random" in env_name:
            kwargs['random_resets'] = False
      elif env_name == "sawyer_dial_turn-goals":
        CLASS = SawyerDialTurnGoalsV2
        max_episode_steps = 150



  elif "ceborl" in env_name:
      # Eg: offline_ceborl-dial_turn
      max_episode_steps = 50
      if "offline" in env_name:
          env_name = env_name[:][len("offline_"):]
          print("env_name:", env_name)

      task_name = env_name.split("-")[-1]
      print("task_name:", task_name)
      gym_env, obs_dim = ceborl_envs.get_env(task_name, image_obs="image" in env_name)

      return gym_env, obs_dim, max_episode_steps

  elif "fetch" in env_name:
      max_episode_steps = 50
      if "offline" in env_name:
          env_name = env_name[:][len("offline_"):]
          print("env_name:", env_name)

      if "fetch_reach" in env_name:
          # kwargs['camera'] = "camera2"
          if "image" in env_name:
              if "goals" in env_name:
                  if "colors" in env_name:
                      CLASS = fetch_envs.FetchReachImageGoalsRandColors
                  if "occluded" in env_name:
                      CLASS = fetch_envs.FetchReachImageGoalsOccluded
                  else:
                      CLASS = fetch_envs.FetchReachImageGoals
              else:
                  CLASS = fetch_envs.FetchReachImage
          else:
              if "goals" in env_name:
                  CLASS = fetch_envs.FetchReachEnvGoals
              else:
                  CLASS = fetch_envs.FetchReachEnv
      elif "fetch_push" in env_name:
          if "image" in env_name:
              kwargs['rand_y'] = True
              if "goals" in env_name:
                  # if "colors" in env_name:
                      # CLASS = fetch_envs.FetchPushImageGoalsRandColors
                  # elif "red" in env_name:
                  #     CLASS = fetch_envs.FetchPushImageGoalsRED
                  # elif "occluded" in env_name:
                  #     CLASS = fetch_envs.FetchPushImageGoalsOccluded
                  if "minimal" in env_name:
                      if "colors" in env_name:
                          CLASS = fetch_envs.FetchPushImageMinimalGoalsColors
                      elif "occluded" in env_name:
                          CLASS = fetch_envs.FetchPushImageMinimalGoalsOccluded
                      else:
                          CLASS = fetch_envs.FetchPushImageMinimalGoals
                  elif "samelocreset" in env_name:
                      CLASS = fetch_envs.FetchPushImageMinimalSameLocResetGoals
                  else:
                      CLASS = fetch_envs.FetchPushImageGoals
              else:
                  CLASS = fetch_envs.FetchPushImage
          else:
              if "push3" in env_name:
                  if "goals" in env_name:
                      CLASS = fetch_envs.FetchPushEnv3Goals
                  else:
                      CLASS = fetch_envs.FetchPushEnv3
              elif "pushsamelocreset" in env_name:
                  if "goals" in env_name:
                      CLASS = fetch_envs.FetchPushEnvSameLocResetGoals
                  else:
                      CLASS = fetch_envs.FetchPushEnvSameLocReset
              else:
                  if "goals" in env_name:
                      CLASS = fetch_envs.FetchPushEnvGoals
                  else:
                      CLASS = fetch_envs.FetchPushEnv

          if "determ" in env_name:
              kwargs['same_block_start_pos'] = True
              kwargs['rand_y'] = False

          if "natg" in env_name:
              kwargs["use_natural_goal"] = True

      elif "fetch_slide" in env_name:
          if "image" in env_name:
              if "goals" in env_name:
                  CLASS = fetch_envs.FetchSlideImageGoals
              else:
                  CLASS = fetch_envs.FetchSlideImage
          else:
              if "goals" in env_name:
                  CLASS = fetch_envs.FetchSlideEnvGoals
              else:
                  CLASS = fetch_envs.FetchSlideEnv
      elif "fetch_pick_and_place" in env_name:
          if "image" in env_name:
              if "goals" in env_name:
                  CLASS = fetch_envs.FetchPickAndPlaceImageGoals
              else:
                  CLASS = fetch_envs.FetchPickAndPlaceImage
          else:
              if "goals" in env_name:
                  CLASS = fetch_envs.FetchPickAndPlaceEnvGoals
              else:
                  CLASS = fetch_envs.FetchPickAndPlaceEnv


      if "push" in env_name or "reach" in env_name:
          if "goals" in env_name and "no-noise" in env_name:
              kwargs["add_goal_noise"] = False
          elif "goals" in env_name:
              kwargs["add_goal_noise"] = True

      if "dense" in env_name:
          kwargs["dense_reward"] = True

  elif env_name.startswith('ant_'):
    _, map_name = env_name.split('_')
    assert map_name in ['umaze', 'medium', 'large']
    CLASS = ant_env.AntMaze
    kwargs['map_name'] = map_name
    kwargs['non_zero_reset'] = True
    if map_name == 'umaze':
      max_episode_steps = 700
    else:
      max_episode_steps = 1000
  elif env_name.startswith('offline_ant'):
    CLASS = lambda: ant_env.make_offline_ant(env_name)
    if 'umaze' in env_name:
      max_episode_steps = 700
    else:
      max_episode_steps = 1000
  elif env_name.startswith('point_image'):
    CLASS = point_env.PointImage
    kwargs['walls'] = env_name.split('_')[-1]
    if '11x11' in env_name:
      max_episode_steps = 100
    else:
      max_episode_steps = 50
  elif env_name.startswith('fixed-goal-point_'): ###===###
    CLASS = point_env_fixed_goal.PointEnvFixedGoal
    kwargs['walls'] = env_name.split('_')[-1]
    if '11x11' in env_name:
      max_episode_steps = 100
    else:
      max_episode_steps = 50 ###---###
  elif env_name.startswith('point_'):
    CLASS = point_env.PointEnv
    kwargs['walls'] = env_name.split('_')[-1]
    if '11x11' in env_name:
      max_episode_steps = 100
    else:
      max_episode_steps = 50
  else:
    raise NotImplementedError('Unsupported environment: %s' % env_name)

  # Disable type checking in line below because different environments have
  # different kwargs, which pytype doesn't reason about.
  gym_env = CLASS(**kwargs)  # pytype: disable=wrong-keyword-args
  obs_dim = gym_env.observation_space.shape[0] // 2
  print("gym_env:", gym_env)
  return gym_env, obs_dim, max_episode_steps



class SawyerPush(metaworld.envs.mujoco.env_dict.ALL_V2_ENVIRONMENTS['push-v2']):
  """Wrapper for the SawyerPush environment."""

  def __init__(self,
               goal_min_x=-0.1,
               goal_min_y=0.5,
               goal_max_x=0.1,
               goal_max_y=0.9):
    super(SawyerPush, self).__init__()
    self._random_reset_space.low[3] = goal_min_x
    self._random_reset_space.low[4] = goal_min_y
    self._random_reset_space.high[3] = goal_max_x
    self._random_reset_space.high[4] = goal_max_y
    self._partially_observable = False
    self._freeze_rand_vec = False
    self._set_task_called = True
    self.reset()
    self._freeze_rand_vec = False  # Set False to randomize the goal position.

  @property
  def observation_space(self):
    return gym.spaces.Box(
        low=np.full(14, -np.inf),
        high=np.full(14, np.inf),
        dtype=np.float32)

  def _get_obs(self):
    finger_right, finger_left = (self._get_site_pos('rightEndEffector'),
                                 self._get_site_pos('leftEndEffector'))
    tcp_center = (finger_right + finger_left) / 2.0
    gripper_distance = np.linalg.norm(finger_right - finger_left)
    gripper_distance = np.clip(gripper_distance / 0.1, 0., 1.)
    obj = self._get_pos_objects()
    # Note: we should ignore the target gripper distance. The arm goal is set
    # to be the same as the puck goal.
    state = np.concatenate([tcp_center, obj, [gripper_distance]])
    goal = np.concatenate([self._target_pos, self._target_pos, [0.5]])
    return np.concatenate([state, goal]).astype(np.float32)

  def step(self, action):
    obs = super(SawyerPush, self).step(action)
    dist = np.linalg.norm(self._target_pos - self._get_pos_objects())
    r = float(dist < 0.05)  # Taken from the metaworld code.
    return obs, r, False, {}



class SawyerPushGoals(SawyerPush):
    def __init__(self,
                 goal_min_x=-0.1,
                 goal_min_y=0.5,
                 goal_max_x=0.1,
                 goal_max_y=0.9):
      super(SawyerPush, self).__init__()
      self._random_reset_space.low[3] = 0
      self._random_reset_space.low[4] = 0
      self._random_reset_space.high[3] = 0
      self._random_reset_space.high[4] = 0
      self._partially_observable = False
      self._freeze_rand_vec = False
      self._set_task_called = True
      self.reset()
      self._freeze_rand_vec = True  # Set False to randomize the goal position.

    def get_expert_goals(self):
      return None

    def _get_orig_obs(self):
        return super(SawyerPush, self)._get_obs()

    def render(self, mode="rgb_array", size=(64, 64), hide_target=True):
        if hide_target:
            self.sim.data.site_xpos[0] = 1_000_000
        img = self.sim.render(size[0], size[1], camera_name="corner2")
        # img = np.flip(img, axis=0)
        return img



class SawyerDrawer(
    metaworld.envs.mujoco.env_dict.ALL_V2_ENVIRONMENTS['drawer-close-v2']):
  """Wrapper for the SawyerDrawer environment."""

  def __init__(self):
    super(SawyerDrawer, self).__init__()
    self._random_reset_space.low[0] = 0
    self._random_reset_space.high[0] = 0
    self._partially_observable = False
    self._freeze_rand_vec = False
    self._set_task_called = True
    self._target_pos = np.zeros(0)  # We will overwrite this later.
    self.reset()
    self._freeze_rand_vec = False  # Set False to randomize the goal position.

  def _get_pos_objects(self):
    return self.get_body_com('drawer_link') +  np.array([.0, -.16, 0.0])

  def reset_model(self):
    super(SawyerDrawer, self).reset_model()
    # First set it to a random location, set as target pos
    self._set_obj_xyz(np.random.uniform(-0.15, 0.0))
    self._target_pos = self._get_pos_objects().copy()

    # Then randomly reset it for real
    self._set_obj_xyz(np.random.uniform(-0.15, 0.0))
    assert False

    return self._get_obs()

  @property
  def observation_space(self):
    return gym.spaces.Box(
        low=np.full(8, -np.inf),
        high=np.full(8, np.inf),
        dtype=np.float32)

  def _get_obs(self):
    finger_right, finger_left = (self._get_site_pos('rightEndEffector'),
                                 self._get_site_pos('leftEndEffector'))
    tcp_center = (finger_right + finger_left) / 2.0
    obj = self._get_pos_objects()
    # Arm position is same as drawer position. We only provide the drawer
    # Y coordinate.
    return np.concatenate([tcp_center, [obj[1]], self._target_pos, [self._target_pos[1]]]).astype(np.float32)

  def step(self, action):
    obs = super(SawyerDrawer, self).step(action)
    return obs, 0.0, False, {}


class SawyerDrawerGoals(SawyerDrawer):
    def reset_model(self):
      super(SawyerDrawer, self).reset_model() # This goes back two layers

      # # First set it to a random location, set as target pos
      # self._set_obj_xyz(np.random.uniform(-0.15, 0.0))
      # self._target_pos = self._get_pos_objects().copy()

      # Then randomly reset it for real
      self._set_obj_xyz(np.random.uniform(-0.15, 0.0))
      # print("self._target_pos:", self._target_pos)
      return self._get_obs()

    def get_expert_goals(self):
        return None

    def step(self, action):
        obs, reward, done, info = super(SawyerDrawerGoals, self).step(action)
        # dist = abs(obs[3] - obs[7])
        # reward = float(dist < 0.02)
        obj = self._get_pos_objects()
        target_pos = obs[4:7]
        dist = np.linalg.norm(obj - target_pos)
        reward = float(dist < 0.01)
        return obs, reward, done, info


class SawyerDialTurnGoalsV2(
    metaworld.envs.mujoco.env_dict.ALL_V2_ENVIRONMENTS['dial-turn-v2']):
    """Wrapper for the SawyerLeverPull environment."""

    def __init__(self):
      super(SawyerDialTurnGoalsV2, self).__init__()
      self._random_reset_space.low[:2] = np.array([-0.4, 0.7]) ###$$$###
      self._random_reset_space.high[:2] = np.array([0.4, 0.8]) ###$$$###

      self._partially_observable = False
      self._freeze_rand_vec = False
      self._set_task_called = True
      self._target_pos = np.zeros(3)  # We will overwrite this later.
      # self._target_pos = np.array([-0.14660674,  0.71726966,  0.45])
      self.reset()
      self._freeze_rand_vec = False  # Set False to randomize the goal position.


    def reset_model(self):
      super(SawyerDialTurnGoalsV2, self).reset_model()
      # self.data.set_joint_qpos('window_slide', np.random.uniform(0.0, 0.2))
      # self._target_pos = self._get_pos_objects().copy()
      self.data.set_joint_qpos('knob_Joint_1', np.random.uniform(0., 0.2)) ###???###
      return self._get_obs()

    @property
    def observation_space(self):
      return gym.spaces.Box(
          low=np.full(12, -np.inf),
          high=np.full(12, np.inf),
          dtype=np.float32)

    def _get_orig_obs(self):
        return super(SawyerDialTurnGoalsV2, self)._get_obs()

    def _get_obs(self):
      finger_right, finger_left = (self._get_site_pos('rightEndEffector'),
                                   self._get_site_pos('leftEndEffector'))
      tcp_center = (finger_right + finger_left) / 2.0
      obj = self._get_pos_objects()
      # print("\nobj:", obj)
      # print("self._target_pos:", self._target_pos)
      return np.concatenate([tcp_center, obj, self._target_pos, self._target_pos]).astype(np.float32)

    def get_expert_goals(self):
        return None

    def step(self, action):
        obs, _, _, _ = super(SawyerDialTurnGoalsV2, self).step(action)
        obj = obs[3:6]
        target_pos = obs[6:9]

        # print("obj:", obj)
        # print("target_pos:", target_pos)
        dist = np.linalg.norm(target_pos - obj)
        # print("dist:", dist)
        reward = float(dist <= 0.07)

        return obs, reward, False, {}

    def render(self, mode="rgb_array", size=(64, 64), hide_target=True):
        if hide_target:
            self.sim.data.site_xpos[0] = 1_000_000
        img = self.sim.render(size[0], size[1], camera_name="corner2")
        # img = np.flip(img, axis=0)
        return img



class SawyerLeverPullGoalsV2(
    metaworld.envs.mujoco.env_dict.ALL_V2_ENVIRONMENTS['lever-pull-v2']):
    """Wrapper for the SawyerLeverPull environment."""

    def __init__(self, random_resets=True):
      self._lever_angle_desired = np.pi / 2.0
      super(SawyerLeverPullGoalsV2, self).__init__()

      if random_resets:
          self._random_reset_space.low[:2] = np.array([-0.4, 0.7])
          self._random_reset_space.high[:2] = np.array([0.4, 0.8])
      else:
          self._random_reset_space.low[:2] = np.array([0, 0.75])
          self._random_reset_space.high[:2] = np.array([0, 0.75])

      self._partially_observable = False
      self._freeze_rand_vec = False
      self._set_task_called = True
      self._target_pos = np.zeros(3)  # We will overwrite this later.
      # self._target_pos = np.array([-0.14660674,  0.71726966,  0.45])
      self.reset()

      if random_resets:
          self._freeze_rand_vec = False  # Set False to randomize the goal position.
      else:
          self._freeze_rand_vec = True


    def reset_model(self):
      super(SawyerLeverPullGoalsV2, self).reset_model()
      # self.data.set_joint_qpos('window_slide', np.random.uniform(0.0, 0.2))
      # self._target_pos = self._get_pos_objects().copy()
      # self.data.set_joint_qpos('LeverAxis', np.random.uniform(0.0, 0.2)) ###???###
      return self._get_obs()

    @property
    def observation_space(self):
      return gym.spaces.Box(
          low=np.full(14, -np.inf),
          high=np.full(14, np.inf),
          dtype=np.float32)

    def _get_orig_obs(self):
        return super(SawyerLeverPullGoalsV2, self)._get_obs()

    def _get_obs(self):
      finger_right, finger_left = (self._get_site_pos('rightEndEffector'),
                                   self._get_site_pos('leftEndEffector'))
      tcp_center = (finger_right + finger_left) / 2.0
      obj = self._get_pos_objects()
      lever_angle = -self.data.get_joint_qpos('LeverAxis')
      lever_angle_desired = self._lever_angle_desired
      # Arm position is same as lever position. Only use Z position of lever.
      # print("\nlever_angle:", lever_angle)
      # print("lever_angle_desired:", lever_angle_desired)
      return np.concatenate([tcp_center, obj, [lever_angle], self._target_pos, self._target_pos, [lever_angle_desired]]).astype(np.float32)

    def get_expert_goals(self):
        return None

    def step(self, action):
        obs, _, _, _ = super(SawyerLeverPullGoalsV2, self).step(action)
        lever_angle = obs[6]
        lever_angle_desired = obs[13]
        # print("lever_angle:", lever_angle)
        # print("lever_angle_desired:", lever_angle_desired)


        lever_error = abs(lever_angle - lever_angle_desired)
        reward = (lever_error <= np.pi / 24)
        # dist = abs(obs[3] - obs[7])
        # reward = float(dist < 0.02)
        return obs, reward, False, {}

    def render(self, mode="rgb_array", size=(64, 64), hide_target=True):
        if hide_target:
            self.sim.data.site_xpos[0] = 1_000_000
        img = self.sim.render(size[0], size[1], camera_name="corner2")
        # img = np.flip(img, axis=0)
        return img




class SawyerWindow(
    metaworld.envs.mujoco.env_dict.ALL_V2_ENVIRONMENTS['window-open-v2']):
  """Wrapper for the SawyerWindow environment."""

  def __init__(self):
    super(SawyerWindow, self).__init__()
    self._random_reset_space.low[:2] = np.array([0.0, 0.8])
    self._random_reset_space.high[:2] = np.array([0.0, 0.8])
    self._partially_observable = False
    self._freeze_rand_vec = False
    self._set_task_called = True
    self._target_pos = np.zeros(3)  # We will overwrite this later.
    self.reset()
    self._freeze_rand_vec = False  # Set False to randomize the goal position.

  def reset_model(self):
    super(SawyerWindow, self).reset_model()
    self.data.set_joint_qpos('window_slide', np.random.uniform(0.0, 0.2))
    self._target_pos = self._get_pos_objects().copy()
    self.data.set_joint_qpos('window_slide', np.random.uniform(0.0, 0.2))
    assert False
    return self._get_obs()

  @property
  def observation_space(self):
    return gym.spaces.Box(
        low=np.full(8, -np.inf),
        high=np.full(8, np.inf),
        dtype=np.float32)

  def _get_obs(self):
    finger_right, finger_left = (self._get_site_pos('rightEndEffector'),
                                 self._get_site_pos('leftEndEffector'))
    tcp_center = (finger_right + finger_left) / 2.0
    obj = self._get_pos_objects()
    # Arm position is same as window position. Only use X position of window.
    return np.concatenate([tcp_center, [obj[0]], self._target_pos, [self._target_pos[0]]]).astype(np.float32)

  def step(self, action):
    obs = super(SawyerWindow, self).step(action)
    return obs, 0.0, False, {}




class SawyerWindowGoals(SawyerWindow):
    def reset_model(self):
      super(SawyerWindow, self).reset_model()
      # self.data.set_joint_qpos('window_slide', np.random.uniform(0.0, 0.2))
      # self._target_pos = self._get_pos_objects().copy()
      self.data.set_joint_qpos('window_slide', np.random.uniform(0.0, 0.2))
      # print("self._target_pos:", self._target_pos)

      # camera_name = 'corner2'
      # index = self.model.camera_name2id(camera_name)
      # self.model.cam_fovy[index] = 17.0
      # self.model.cam_pos[index][1] = -0.1
      # self.model.cam_pos[index][2] = 1.1

      return self._get_obs()

    def get_expert_goals(self):
        return None

    def step(self, action):
        obs, reward, done, info = super(SawyerWindowGoals, self).step(action)
        dist = abs(obs[3] - obs[7])
        # reward = float(dist < 0.02)
        reward = float(dist < 0.05)
        return obs, reward, done, info

    def render(self, mode="rgb_array", height=64, width=64):
        for ctx in self.sim.render_contexts:
          ctx.opengl_context.make_context_current()

        # self.sim.data.site_xpos[0] = 1_000_000
        img = self.sim.render(height, width, camera_name="corner2")
        # img = np.flip(img, axis=0)
        return img

    def _sample_goal(self):
        pass


class SawyerWindowImageMinimalGoals(SawyerWindowGoals):
    def __init__(self, *args, **kwargs):
        super(SawyerWindowImageMinimalGoals, self).__init__(*args, **kwargs)
        camera_name = 'corner2'
        index = self.model.camera_name2id(camera_name)
        self.model.cam_fovy[index] = 17.0
        self.model.cam_pos[index][1] = -0.1
        self.model.cam_pos[index][2] = 1.1

    @property
    def observation_space(self):
      return gym.spaces.Box(
          low=np.full((64*64*6), 0),
          high=np.full((64*64*6), 255),
          dtype=np.uint8)

    def image_obs(self):
        self.sim.data.site_xpos[0] = 1_000_000
        img = self.render(mode='rgb_array', height=64, width=64)
        return img.flatten()

    def reset_model(self):
        super(SawyerWindowImageMinimalGoals, self).reset_model()
        img = self.image_obs()
        self._goal_img = np.zeros_like(img)
        return np.concatenate([img, np.zeros_like(img)])

    def step(self, *args, **kwargs):
        obs, reward, done, info = super(SawyerWindowImageMinimalGoals, self).step(*args, **kwargs)
        img = self.image_obs()
        return np.concatenate([img, np.zeros_like(img)]), reward, done, info



class SawyerBin(
    metaworld.envs.mujoco.env_dict.ALL_V2_ENVIRONMENTS['bin-picking-v2']):
  """Wrapper for the SawyerBin environment."""

  def __init__(self):
    self._goal = np.zeros(3)
    super(SawyerBin, self).__init__()
    self._partially_observable = False
    self._freeze_rand_vec = False
    self._set_task_called = True
    self.reset()
    self._freeze_rand_vec = False  # Set False to randomize the goal position.

  def reset(self):
    super(SawyerBin, self).reset()
    body_id = self.model.body_name2id('bin_goal')
    pos1 = self.sim.data.body_xpos[body_id].copy()
    pos1 += np.random.uniform(-0.05, 0.05, 3)
    pos2 = self._get_pos_objects().copy()
    t = np.random.random()
    self._goal = t * pos1 + (1 - t) * pos2
    self._goal[2] = np.random.uniform(0.03, 0.12)
    return self._get_obs()

  def step(self, action):
    super(SawyerBin, self).step(action)
    dist = np.linalg.norm(self._goal - self._get_pos_objects())
    r = float(dist < 0.05)  # Taken from metaworld
    done = False
    info = {}
    return self._get_obs(), r, done, info

  def _get_obs(self):
    pos_hand = self.get_endeff_pos()
    finger_right, finger_left = (
        self._get_site_pos('rightEndEffector'),
        self._get_site_pos('leftEndEffector')
    )
    gripper_distance_apart = np.linalg.norm(finger_right - finger_left)
    gripper_distance_apart = np.clip(gripper_distance_apart / 0.1, 0., 1.)
    obs = np.concatenate((pos_hand, [gripper_distance_apart],
                          self._get_pos_objects()))
    goal = np.concatenate([self._goal + np.array([0.0, 0.0, 0.03]),
                           [0.4], self._goal])
    return np.concatenate([obs, goal]).astype(np.float32)

  @property
  def observation_space(self):
    return gym.spaces.Box(
        low=np.full(2 * 7, -np.inf),
        high=np.full(2 * 7, np.inf),
        dtype=np.float32)


class SawyerBinGoals(
    metaworld.envs.mujoco.env_dict.ALL_V2_ENVIRONMENTS['bin-picking-v2']):
  """Wrapper for the SawyerBin environment."""

  def __init__(self):
    self._goal = np.zeros(3)
    super(SawyerBinGoals, self).__init__()
    self._partially_observable = False
    self._freeze_rand_vec = False
    self._set_task_called = True
    self.reset()
    self._freeze_rand_vec = False  # Set False to randomize the goal position.

  def reset(self):
    # super(SawyerBinGoals, self).reset()
    # body_id = self.model.body_name2id('bin_goal')
    # pos1 = self.sim.data.body_xpos[body_id].copy()
    # pos1 += np.random.uniform(-0.05, 0.05, 3)
    # pos2 = self._get_pos_objects().copy()
    # t = np.random.random()
    # self._goal = t * pos1 + (1 - t) * pos2
    # self._goal[2] = np.random.uniform(0.03, 0.12)
    # return self._get_obs()

    super(SawyerBinGoals, self).reset()
    # body_id = self.model.body_name2id('bin_goal')
    # pos1 = self.sim.data.body_xpos[body_id].copy()
    # pos1 += np.random.uniform(-0.05, 0.05, 3)
    # pos2 = self._get_pos_objects().copy()
    # t = np.random.random()
    # self._goal = t * pos1 + (1 - t) * pos2
    # self._goal[2] = np.random.uniform(0.03, 0.12)
    self._goal = np.array([0.12, 0.7, 0.02])
    return self._get_obs()

  def step(self, action):
    super(SawyerBinGoals, self).step(action)
    dist = np.linalg.norm(self._goal - self._get_pos_objects())
    r = float(dist < 0.05)  # Taken from metaworld
    done = False
    info = {}
    return self._get_obs(), r, done, info

  def _get_obs(self):
    pos_hand = self.get_endeff_pos()
    finger_right, finger_left = (
        self._get_site_pos('rightEndEffector'),
        self._get_site_pos('leftEndEffector')
    )
    gripper_distance_apart = np.linalg.norm(finger_right - finger_left)
    gripper_distance_apart = np.clip(gripper_distance_apart / 0.1, 0., 1.)
    obs = np.concatenate((pos_hand, [gripper_distance_apart],
                          self._get_pos_objects()))
    goal = np.concatenate([self._goal + np.array([0.0, 0.0, 0.03]),
                           [0.4], self._goal])
    return np.concatenate([obs, goal]).astype(np.float32)

  @property
  def observation_space(self):
    return gym.spaces.Box(
        low=np.full(2 * 7, -np.inf),
        high=np.full(2 * 7, np.inf),
        dtype=np.float32)

  def get_expert_goals(self):
      return None

# class SawyerBinGoals(SawyerBin):
#     def reset_model(self):
      # super(SawyerBinGoals, self).reset()
      # # body_id = self.model.body_name2id('bin_goal')
      # # pos1 = self.sim.data.body_xpos[body_id].copy()
      # # pos1 += np.random.uniform(-0.05, 0.05, 3)
      # # pos2 = self._get_pos_objects().copy()
      # # t = np.random.random()
      # # self._goal = t * pos1 + (1 - t) * pos2
      # # self._goal[2] = np.random.uniform(0.03, 0.12)
      # self._goal = np.array([0.12, 0.7, 0.02])
      # return self._get_obs()



class SawyerDrawerImage(SawyerDrawer):
  """Wrapper for the SawyerDrawer environment with image observations."""

  def __init__(self, camera='corner2', task='openclose'):
    self._task = task
    self._camera_name = camera
    self._dist = []
    self._dist_vec = []
    super(SawyerDrawerImage, self).__init__()

  def reset_metrics(self):
    self._dist_vec = []
    self._dist = []

  def step(self, action):
    _, _, done, info = super(SawyerDrawerImage, self).step(action)
    y = self._get_pos_objects()[1]
    # L1 distance between current and target drawer location.
    dist = abs(y - self._goal_y)
    self._dist.append(dist)
    r = float(dist < 0.04)
    img = self._get_img()
    return np.concatenate([img, self._goal_img], axis=-1), r, done, info

  def _move_hand_to_obj(self):
    for _ in range(20):
      self.data.set_mocap_pos(
          'mocap', self._get_pos_objects() + np.array([0.0, 0.0, 0.03]))
      self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
      self.do_simulation([-1, 1], self.frame_skip)

  def reset(self):
    if self._dist:
      self._dist_vec.append(self._dist)
    self._dist = []

    # reset the cameras
    camera_name = 'behindGripper'
    index = self.model.camera_name2id(camera_name)
    self.model.cam_fovy[index] = 30.0

    camera_name = 'topview'
    index = self.model.camera_name2id(camera_name)
    self.model.cam_fovy[index] = 20.0
    self.model.cam_pos[index][1] = 0.7

    camera_name = 'corner2'
    index = self.model.camera_name2id(camera_name)
    self.model.cam_fovy[index] = 8.0
    self.model.cam_pos[index][0] = 1.5
    self.model.cam_pos[index][1] = -0.2
    self.model.cam_pos[index][2] = 1.1

    camera_name = 'corner3'
    index = self.model.camera_name2id(camera_name)
    self.model.cam_fovy[index] = 30.0
    self.model.cam_pos[index][0] = 0.3
    self.model.cam_pos[index][1] = 0.45
    self.model.cam_pos[index][2] = 0.7

    # Get the goal image.
    super(SawyerDrawerImage, self).reset()
    self._move_hand_to_obj()
    self._goal_y = self._get_pos_objects()[1]
    self._goal_img = self._get_img()

    # Reset the environment again.
    super(SawyerDrawerImage, self).reset()
    if self._task == 'close':
      self._set_obj_xyz(-0.15)
    elif self._task == 'open':
      self._set_obj_xyz(0.0)
    else:
      assert self._task == 'openclose'
      self._set_obj_xyz(np.random.choice([-0.15, 0.0]))
    self._move_hand_to_obj()
    img = self._get_img()

    # Add the initial distance.
    y = self._get_pos_objects()[1]
    # L1 distance between current and target drawer location.
    dist = abs(y - self._goal_y)
    self._dist.append(dist)
    return np.concatenate([img, self._goal_img], axis=-1)

  def _get_img(self):
    assert self._camera_name in ['behindGripper', 'topview',
                                 'corner2', 'corner3']
    # Hide the goal marker position
    self._set_pos_site('goal', np.inf * self._target_pos)
    # IMPORTANT: Pull the context to the current thread.
    for ctx in self.sim.render_contexts:
      ctx.opengl_context.make_context_current()

    img = self.render(offscreen=True,
                      resolution=(64, 64),
                      camera_name=self._camera_name)
    if self._camera_name in ['behindGripper']:
      img = img[::-1]
    return img.flatten()

  @property
  def observation_space(self):
    return gym.spaces.Box(
        low=np.full((64*64*6), 0),
        high=np.full((64*64*6), 255),
        dtype=np.uint8)


class SawyerPushImage(
    metaworld.envs.mujoco.env_dict.ALL_V2_ENVIRONMENTS['push-v2']):
  """Wrapper for the SawyerPush environment with image observations."""

  def __init__(self, camera='corner2', rand_y=True, start_at_obj=False):
    self._start_at_obj = start_at_obj
    self._rand_y = rand_y
    self._camera_name = camera
    self._dist = []
    self._dist_vec = []
    super(SawyerPushImage, self).__init__()
    self._partially_observable = False
    self._freeze_rand_vec = False
    self._set_task_called = True
    self.reset()
    self._freeze_rand_vec = False  # Set False to randomize the goal position.

  def reset(self):
    if self._dist:
      self._dist_vec.append(self._dist)
    self._dist = []

    camera_name = 'corner'
    index = self.model.camera_name2id(camera_name)
    self.model.cam_fovy[index] = 20.0
    self.model.cam_pos[index][2] = 0.5
    self.model.cam_pos[index][0] = -1.0

    camera_name = 'corner2'
    index = self.model.camera_name2id(camera_name)
    self.model.cam_fovy[index] = 45
    self.model.cam_pos[index][0] = 0.7
    self.model.cam_pos[index][1] = 0.65
    self.model.cam_pos[index][2] = 0.1
    self.model.cam_quat[index] = euler2quat(
        np.array([-np.pi / 2, np.pi / 2, 0.0]))

    # Get the goal image.
    s = super(SawyerPushImage, self).reset()
    self._goal = s[:7][3:6]
    self._goal[1] += np.random.uniform(0.0, 0.25)
    if self._rand_y:
      self._goal[0] += np.random.uniform(-0.1, 0.1)
    self._set_obj_xyz(self._goal)
    for _ in range(200):
      self.data.set_mocap_pos('mocap', self._get_pos_objects())
      self._set_obj_xyz(self._goal)
      self.do_simulation([-1, 1], self.frame_skip)
    self._goal_img = self._get_img()

    # Reset the environment again.
    s = super(SawyerPushImage, self).reset()
    obj = s[:7][3:6] + np.array([0.0, -0.2, 0.0])
    self._set_obj_xyz(obj)
    self.do_simulation([-1, 1], self.frame_skip)
    if self._start_at_obj:
      for _ in range(20):
        self.data.set_mocap_pos('mocap', self._get_pos_objects())
        self.do_simulation([-1, 1], self.frame_skip)
    img = self._get_img()

    # Add the first distances
    obj = self.get_body_com('obj')
    dist = np.linalg.norm(obj - self._goal)
    self._dist.append(dist)
    return np.concatenate([img, self._goal_img], axis=-1)

  def step(self, action):
    super(SawyerPushImage, self).step(action)
    obj = self.get_body_com('obj')
    dist = np.linalg.norm(obj - self._goal)
    r = float(dist < 0.05)  # Taken from the metaworld code.
    self._dist.append(dist)
    img = self._get_img()
    done = False
    info = {}
    return np.concatenate([img, self._goal_img], axis=-1), r, done, info

  def _get_img(self):
    if self._camera_name.startswith('default-'):
      camera_name = self._camera_name.split('default-')[1]
    else:
      camera_name = self._camera_name
    # Hide the goal marker position.
    self._set_pos_site('goal', np.inf * self._target_pos)
    # IMPORTANT: Pull the context to the current thread.
    for ctx in self.sim.render_contexts:
      ctx.opengl_context.make_context_current()
    img = self.render(offscreen=True, resolution=(64, 64),
                      camera_name=camera_name)
    if camera_name in ['behindGripper']:
      img = img[::-1]
    return img.flatten()

  @property
  def observation_space(self):
    return gym.spaces.Box(
        low=np.full((64*64*6), 0),
        high=np.full((64*64*6), 255),
        dtype=np.uint8)


class SawyerWindowImage(SawyerWindow):
  """Wrapper for the SawyerWindow environment with image observations."""

  def __init__(self, task=None, start_at_obj=True):
    self._start_at_obj = start_at_obj
    self._task = task
    self._camera_name = 'corner2'
    self._dist = []
    self._dist_vec = []
    super(SawyerWindowImage, self).__init__()

  def reset_metrics(self):
    self._dist_vec = []
    self._dist = []

  def step(self, action):
    _, _, done, info = super(SawyerWindowImage, self).step(action)
    x = self.data.get_joint_qpos('window_slide')
    # L1 distance between current and target drawer location.
    dist = abs(x - self._goal_x)
    self._dist.append(dist)
    r = (dist < 0.05)
    img = self._get_img()
    return np.concatenate([img, self._goal_img], axis=-1), r, done, info

  def reset(self):
    if self._dist:
      self._dist_vec.append(self._dist)
    self._dist = []

    # Reset the cameras.
    camera_name = 'corner2'
    index = self.model.camera_name2id(camera_name)
    if self._start_at_obj:
      self.model.cam_fovy[index] = 10.0
      self.model.cam_pos[index][0] = 1.5
      self.model.cam_pos[index][1] = -0.1
      self.model.cam_pos[index][2] = 1.1
    else:
      self.model.cam_fovy[index] = 17.0
      self.model.cam_pos[index][1] = -0.1
      self.model.cam_pos[index][2] = 1.1

    # Get the goal image.
    super(SawyerWindowImage, self).reset()
    goal_slide_pos = np.random.uniform(0, 0.2)
    for _ in range(20):
      self.data.set_mocap_pos('mocap', self._get_pos_objects())
      self.data.set_joint_qpos('window_slide', goal_slide_pos)
      self.do_simulation([-1, 1], self.frame_skip)
    self._goal_x = goal_slide_pos
    self._goal_img = self._get_img()

    # Reset the environment again.
    super(SawyerWindowImage, self).reset()
    if self._task == 'open':
      init_slide_pos = 0.0
    elif self._task == 'close':
      init_slide_pos = 0.2
    else:
      assert self._task == 'openclose'
      init_slide_pos = np.random.choice([0.0, 0.2])

    if self._start_at_obj:
      for _ in range(50):
        self.data.set_mocap_pos('mocap', self._get_pos_objects())
        self.data.set_joint_qpos('window_slide', init_slide_pos)
        self.do_simulation([-1, 1], self.frame_skip)
    else:
      self.data.set_joint_qpos('window_slide', init_slide_pos)
      self.do_simulation([-1, 1], self.frame_skip)
    img = self._get_img()

    # Add the initial distance.
    x = self.data.get_joint_qpos('window_slide')
    # L1 distance between current and target drawer location.
    dist = abs(x - self._goal_x)
    self._dist.append(dist)
    return np.concatenate([img, self._goal_img], axis=-1)

  def _get_img(self):
    assert self._camera_name in ['corner', 'topview', 'corner3',
                                 'behindGripper', 'corner2']
    # Hide the goal marker position.
    self._set_pos_site('goal', np.inf * self._target_pos)
    # IMPORTANT: Pull the context to the current thread.
    for ctx in self.sim.render_contexts:
      ctx.opengl_context.make_context_current()
    img = self.render(offscreen=True,
                      resolution=(64, 64),
                      camera_name=self._camera_name)
    if self._camera_name in ['corner', 'topview', 'behindGripper']:
      img = img[::-1]
    return img.flatten()

  @property
  def observation_space(self):
    return gym.spaces.Box(
        low=np.full((64*64*6), 0),
        high=np.full((64*64*6), 255),
        dtype=np.uint8)


class SawyerBinImage(
    metaworld.envs.mujoco.env_dict.ALL_V2_ENVIRONMENTS['bin-picking-v2']):
  """Wrapper for the SawyerBin environment with image observations."""

  def __init__(self, camera='corner2', start_at_obj=True, alias=False):
    self._alias = alias
    self._start_at_obj = start_at_obj
    self._dist = []
    self._dist_vec = []
    self._camera_name = camera
    super(SawyerBinImage, self).__init__()
    self._partially_observable = False
    self._freeze_rand_vec = False
    self._set_task_called = True
    self.reset()
    self._freeze_rand_vec = False  # Set False to randomize the goal position.

  def reset_metrics(self):
    self._dist_vec = []
    self._dist = []

  def _hand_obj_dist(self):
    body_id = self.model.body_name2id('hand')
    hand_pos = self.sim.data.body_xpos[body_id]
    obj_pos = self._get_pos_objects()
    return np.linalg.norm(hand_pos - obj_pos)

  def _obj_goal_dist(self):
    obj_pos = self._get_pos_objects()
    return np.linalg.norm(self._goal[:2] - obj_pos[:2])

  def step(self, action):
    super(SawyerBinImage, self).step(action)
    dist = self._obj_goal_dist()
    self._dist.append(dist)
    r = float(dist < 0.05)  # Success if within 5cm of the goal.
    img = self._get_img()
    done = False
    info = {}
    return np.concatenate([img, self._goal_img], axis=-1), r, done, info

  def reset(self):
    if self._dist:
      self._dist_vec.append(self._dist)
    self._dist = []

    # reset the cameras
    camera_name = 'corner2'
    index = self.model.camera_name2id(camera_name)
    self.model.cam_fovy[index] = 14.0
    self.model.cam_pos[index][0] = 1.3
    self.model.cam_pos[index][1] = -0.05
    self.model.cam_pos[index][2] = 0.9

    camera_name = 'topview'
    index = self.model.camera_name2id(camera_name)
    self.model.cam_pos[index][1] = 0.7
    self.model.cam_pos[index][2] = 0.9

    # Get the goal image.
    super(SawyerBinImage, self).reset()
    body_id = self.model.body_name2id('bin_goal')
    obj_pos = self.sim.data.body_xpos[body_id].copy()
    obj_pos[:2] += np.random.uniform(-0.05, 0.05, 2)
    obj_pos[2] = 0.05
    self._set_obj_xyz(obj_pos)
    hand_offset = np.random.uniform([0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.05])
    for t in range(40):
      self.data.set_mocap_pos('mocap', obj_pos + hand_offset)
      self.do_simulation((t > 20) * np.array([1.0, -1.0]), self.frame_skip)
    self._goal = self._get_pos_objects().copy()
    self._goal_img = self._get_img()

    # Reset the environment again.
    super(SawyerBinImage, self).reset()
    obj_pos = self._get_pos_objects()
    if self._start_at_obj:
      for t in range(40):
        self.data.set_mocap_pos('mocap', obj_pos + np.array([0.0, 0.0, 0.05]))
        self.do_simulation((t > 40) * np.array([1.0, -1.0]), self.frame_skip)
    img = self._get_img()

    # Add the initial distance.
    self._dist.append(self._obj_goal_dist())
    return np.concatenate([img, self._goal_img], axis=-1)

  def _get_img(self):
    if self._camera_name.startswith('default-'):
      camera_name = self._camera_name.split('default-')[1]
    else:
      camera_name = self._camera_name
    assert camera_name in ['corner', 'topview', 'corner3',
                           'behindGripper', 'corner2']
    # IMPORTANT: Pull the context to the current thread.
    for ctx in self.sim.render_contexts:
      ctx.opengl_context.make_context_current()
    resolution = (64, 64)
    img = self.render(offscreen=True, resolution=resolution,
                      camera_name=camera_name)
    if camera_name in ['corner', 'topview', 'behindGripper']:
      img = img[::-1]
    return img.flatten()

  @property
  def observation_space(self):
    return gym.spaces.Box(
        low=np.full((64*64*6), 0),
        high=np.full((64*64*6), 255),
        dtype=np.uint8)
