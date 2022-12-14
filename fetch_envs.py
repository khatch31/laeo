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

"""Utility for loading the OpenAI Gym Fetch robotics environments."""

import gym
from gym.envs.robotics.fetch import push
from gym.envs.robotics.fetch import reach
from gym.envs.robotics.fetch import slide
from gym.envs.robotics.fetch import pick_and_place
import numpy as np


class ObsDictWrapper:
    def __init__(self, env):
        self._env = env

    def __getattr__(self, attr):
        if attr == '_wrapped_env':
            raise AttributeError()
        return getattr(self._env, attr)

class FetchSlideEnv(slide.FetchSlideEnv):
    def __init__(self, dense_reward=False):
      super(FetchSlideEnv, self).__init__()
      self._old_observation_space = self.observation_space
      self._new_observation_space = gym.spaces.Box(
          low=np.full((50,), -np.inf),
          high=np.full((50,), np.inf),
          dtype=np.float32)
      self.observation_space = self._new_observation_space

      self._dense_reward = dense_reward

    def reset(self):
      self.observation_space = self._old_observation_space
      s = super(FetchSlideEnv, self).reset()
      self.observation_space = self._new_observation_space
      return self.observation(s)

    def step(self, action):
      # s, _, _, _ = super(FetchSlideEnv, self).step(action)
      s, r, _, info = super(FetchSlideEnv, self).step(action)
      done = False
      dist = np.linalg.norm(s['achieved_goal'] - s['desired_goal'])
      if self._dense_reward:
          info = {"success":float(dist < 0.05)}
      else:
          r = float(dist < 0.05)  # Default from Fetch environment.
          info = {}
      return self.observation(s), r, done, info

    def observation(self, observation):
        start_index = 3
        end_index = 6
        goal_pos_1 = observation['achieved_goal']
        goal_pos_2 = observation['observation'][start_index:end_index]
        assert np.all(goal_pos_1 == goal_pos_2)
        s = observation['observation']
        g = np.zeros_like(s)
        g[start_index:end_index] = observation['desired_goal']
        # print(f"\ns.shape: {s.shape}, g.shape: {g.shape}")
        # print(f"s {s}, g: {g}")
        return np.concatenate([s, g]).astype(np.float32)

class FetchSlideEnvGoals(FetchSlideEnv):
    def _sample_goal(self):
        goal = np.array([1.6, 0.9, 0.41401894])
        return goal

    def get_expert_goals(self):
        return None


class FetchPickAndPlaceEnv(pick_and_place.FetchPickAndPlaceEnv):
    def __init__(self, dense_reward=False):
      super(FetchPickAndPlaceEnv, self).__init__()
      self._old_observation_space = self.observation_space
      self._new_observation_space = gym.spaces.Box(
          low=np.full((50,), -np.inf),
          high=np.full((50,), np.inf),
          dtype=np.float32)
      self.observation_space = self._new_observation_space

      self._dense_reward = dense_reward

    def reset(self):
      self.observation_space = self._old_observation_space
      s = super(FetchPickAndPlaceEnv, self).reset()
      self.observation_space = self._new_observation_space
      return self.observation(s)

    def step(self, action):
      # s, _, _, _ = super(FetchPickAndPlaceEnv, self).step(action)
      s, r, _, info = super(FetchPickAndPlaceEnv, self).step(action)
      done = False
      dist = np.linalg.norm(s['achieved_goal'] - s['desired_goal'])
      if self._dense_reward:
          info = {"success":float(dist < 0.05)}
      else:
          r = float(dist < 0.05)  # Default from Fetch environment.
          info = {}

      return self.observation(s), r, done, info

    def observation(self, observation):
        start_index = 3
        end_index = 6
        goal_pos_1 = observation['achieved_goal']
        goal_pos_2 = observation['observation'][start_index:end_index]
        assert np.all(goal_pos_1 == goal_pos_2)
        s = observation['observation']
        g = np.zeros_like(s)
        g[start_index:end_index] = observation['desired_goal']
        # print(f"\ns.shape: {s.shape}, g.shape: {g.shape}")
        # print(f"s {s}, g: {g}")
        return np.concatenate([s, g]).astype(np.float32)


class FetchPickAndPlaceEnvGoals(FetchPickAndPlaceEnv):
    def _sample_goal(self):
        goal = np.array([1.4, 0.7, 0.85])
        return goal

    def get_expert_goals(self):
        return None

class FetchReachEnv(reach.FetchReachEnv):
  """Wrapper for the FetchReach environment."""

  def __init__(self):
    super(FetchReachEnv, self).__init__()
    self._old_observation_space = self.observation_space
    self._new_observation_space = gym.spaces.Box(
        low=np.full((20,), -np.inf),
        high=np.full((20,), np.inf),
        dtype=np.float32)
    self.observation_space = self._new_observation_space


  def reset(self):
    self.observation_space = self._old_observation_space
    s = super(FetchReachEnv, self).reset()
    self.observation_space = self._new_observation_space
    return self.observation(s)

  def step(self, action):
    s, _, _, _ = super(FetchReachEnv, self).step(action)
    done = False
    dist = np.linalg.norm(s['achieved_goal'] - s['desired_goal'])
    r = float(dist < 0.05)  # Default from Fetch environment.
    info = {}
    return self.observation(s), r, done, info

  def observation(self, observation):
    start_index = 0
    end_index = 3
    goal_pos_1 = observation['achieved_goal']
    goal_pos_2 = observation['observation'][start_index:end_index]
    assert np.all(goal_pos_1 == goal_pos_2)
    s = observation['observation']
    g = np.zeros_like(s)
    g[start_index:end_index] = observation['desired_goal']
    # print(f"\ns.shape: {s.shape}, g.shape: {g.shape}")
    # print(f"s {s}, g: {g}")
    return np.concatenate([s, g]).astype(np.float32)

class FetchReachEnvGoals(FetchReachEnv):
  def __init__(self, add_goal_noise=False):
      self._add_goal_noise = add_goal_noise
      super(FetchReachEnvGoals, self).__init__()

  def _sample_goal(self): ###===### ###---###
      # return np.array([1.4, 0.8, 0.6])
      goal =  np.array([1.3, 0.3, 0.9]) #, 0, 0, -5.9625151e-04, -3.4385541e-04, 4.1548879e-04, 1.5108634e-04, 2.9286076e-07])
      if self._add_goal_noise:
          goal += np.random.normal(scale=0.01, size=goal.shape)
      return goal

  def get_expert_goals(self):
      # goals = np.zeros((10, 10))
      # goals[:, 3:] = np.array([0, 0, -5.9625151e-04, -3.4385541e-04, 4.1548879e-04, 1.5108634e-04, 2.9286076e-07])
      # goals[:, :3] = self._sample_goal()
      # return goals
      return None

class FetchPushImageMinimal(push.FetchPushEnv):
  def __init__(self, camera='camera2', start_at_obj=True, rand_y=False, same_block_start_pos=False):
    self._camera_name = camera

    super(FetchPushImageMinimal, self).__init__()
    # self._old_observation_space = self.observation_space
    # self._new_observation_space = gym.spaces.Box(
    #     low=np.full((50,), -np.inf),
    #     high=np.full((50,), np.inf),
    #     dtype=np.float32)
    # self.observation_space = self._new_observation_space
    self._old_observation_space = self.observation_space
    self._new_observation_space = gym.spaces.Box(
        low=np.full((64*64*6), 0),
        high=np.full((64*64*6), 255),
        dtype=np.uint8)
    self.observation_space = self._new_observation_space
    self.sim.model.geom_rgba[1:5] = 0  # Hide the lasers

  def reset(self):
    self.observation_space = self._old_observation_space
    s = super(FetchPushImageMinimal, self).reset()
    self.observation_space = self._new_observation_space
    # return self.observation(s)
    # dist = np.linalg.norm(block_xyz[:2] - self._goal)
    # self._dist.append(dist)
    # if block_xyz[2] < 0.4:  # If block has fallen off the table, recurse.
    #   print('Bad reset, recursing.')
    #   return self.reset()

    img = self.image_obs()
    self._goal_img = np.zeros_like(img)
    return np.concatenate([img, np.zeros_like(img)])

  def step(self, action):
    s, _, _, _ = super(FetchPushImageMinimal, self).step(action)
    done = False
    dist = np.linalg.norm(s['achieved_goal'] - s['desired_goal'])
    r = float(dist < 0.05)
    info = {}
    # info = dict(state=self.observation(s))
    # return self.observation(s), r, done, info

    img = self.image_obs()
    return np.concatenate([img, np.zeros_like(img)]), r, done, info

  def observation(self, observation):
    start_index = 3
    end_index = 6
    goal_pos_1 = observation['achieved_goal']
    goal_pos_2 = observation['observation'][start_index:end_index]
    assert np.all(goal_pos_1 == goal_pos_2)
    s = observation['observation']
    g = np.zeros_like(s)
    g[:start_index] = observation['desired_goal']
    g[start_index:end_index] = observation['desired_goal']
    return np.concatenate([s, g]).astype(np.float32)

  def image_obs(self):
      self.sim.data.site_xpos[0] = 1_000_000
      img = self.render(mode='rgb_array', height=64, width=64)
      return img.flatten()

  def _viewer_setup(self):
    super(FetchPushImageMinimal, self)._viewer_setup()
    if self._camera_name == 'camera1':
      self.viewer.cam.lookat[Ellipsis] = np.array([1.2, 0.8, 0.4])
      self.viewer.cam.distance = 0.9
      self.viewer.cam.azimuth = 180
      self.viewer.cam.elevation = -40
    elif self._camera_name == 'camera2':
      self.viewer.cam.lookat[Ellipsis] = np.array([1.25, 0.8, 0.4])
      self.viewer.cam.distance = 0.65
      self.viewer.cam.azimuth = 90
      self.viewer.cam.elevation = -40
    elif self._camera_name == 'camera3':
      self.viewer.cam.lookat[Ellipsis] = np.array([1.25, 0.8, 0.4])
      self.viewer.cam.distance = 0.9
      self.viewer.cam.azimuth = 90
      self.viewer.cam.elevation = -40
    else:
      raise NotImplementedError

class FetchPushImageMinimalGoals(FetchPushImageMinimal):
    def __init__(self, *args, add_goal_noise=False, **kwargs):
        self._add_goal_noise = add_goal_noise
        super(FetchPushImageMinimalGoals, self).__init__(*args, **kwargs)

    def _sample_goal(self):
        goal = np.array([1.4, 0.9, 0.42469975])
        if self._add_goal_noise:
            goal += np.random.normal(scale=0.01, size=goal.shape)
        return goal

    def get_expert_goals(self):
        return None

class FetchPushImageMinimalSameLocReset(FetchPushImageMinimal):
    def reset(self):
      self.observation_space = self._old_observation_space
      super(FetchPushImageMinimalSameLocReset, self).reset()
      self.observation_space = self._new_observation_space

      object_qpos = np.array([1.21847399e+00,
                              6.17473012e-01,
                              4.24702091e-01,
                              1.00000000e+00,
                              -1.92607042e-07,
                              2.96318526e-07,
                              -2.88376574e-16])

      self.sim.data.set_joint_qpos('object0:joint', object_qpos)

      for _ in range(10):
        super(FetchPushImageMinimalSameLocReset, self).step(np.array([0.0, 0.0, 0.0, 0.0]))

      img = self.image_obs()
      return np.concatenate([img, np.zeros_like(img)])

class FetchPushImageMinimalSameLocResetGoals(FetchPushImageMinimalSameLocReset):
    def __init__(self, *args, add_goal_noise=False, **kwargs):
        self._add_goal_noise = add_goal_noise
        super(FetchPushImageMinimalSameLocResetGoals, self).__init__(*args, **kwargs)

    def _sample_goal(self):
        goal = np.array([1.4, 0.9, 0.42469975])
        if self._add_goal_noise:
            goal += np.random.normal(scale=0.01, size=goal.shape)
        return goal

    def get_expert_goals(self):
        return None

class FetchPushEnv(push.FetchPushEnv):
  """Wrapper for the FetchPush environment."""

  def __init__(self, use_natural_goal=False):
    super(FetchPushEnv, self).__init__()
    self._old_observation_space = self.observation_space
    self._new_observation_space = gym.spaces.Box(
        low=np.full((50,), -np.inf),
        high=np.full((50,), np.inf),
        dtype=np.float32)
    self.observation_space = self._new_observation_space
    self._use_natural_goal = use_natural_goal

  def reset(self):
    self.observation_space = self._old_observation_space
    s = super(FetchPushEnv, self).reset()
    self.observation_space = self._new_observation_space
    return self.observation(s)

  def step(self, action):
    s, _, _, _ = super(FetchPushEnv, self).step(action)
    done = False
    dist = np.linalg.norm(s['achieved_goal'] - s['desired_goal'])
    r = float(dist < 0.05)
    info = {}
    # info = dict(state=self.observation(s))
    return self.observation(s), r, done, info

  def observation(self, observation):
    start_index = 3
    end_index = 6
    goal_pos_1 = observation['achieved_goal']
    goal_pos_2 = observation['observation'][start_index:end_index]
    assert np.all(goal_pos_1 == goal_pos_2)
    s = observation['observation']
    g = np.zeros_like(s)

    if self._use_natural_goal:
        g = np.array([1.34207559e+00,  9.35664892e-01,  4.19158936e-01,  1.38074136e+00,
                  9.01960373e-01,  4.24771219e-01,  3.86658683e-02, -3.37044820e-02,
                  5.61230257e-03,  0.00000000e+00,  0.00000000e+00,  8.61858207e-05,
                 -2.82435009e-04, -1.29339844e-01, -6.43660547e-04, -2.83353089e-04,
                  1.36204704e-04, -4.84256343e-05,  1.58692084e-04,  1.01813814e-13,
                  6.47627865e-04,  2.84563721e-04, -1.28721018e-04,  8.95160338e-06,
                  2.86602793e-04])

    g[:start_index] = observation['desired_goal']
    g[start_index:end_index] = observation['desired_goal']
    return np.concatenate([s, g]).astype(np.float32)

class FetchPushEnvGoals(FetchPushEnv):
    def __init__(self, *args, add_goal_noise=False, **kwargs):
        self._add_goal_noise = add_goal_noise
        super(FetchPushEnvGoals, self).__init__(*args, **kwargs)

    def _sample_goal(self):
        goal = np.array([1.4, 0.9, 0.42469975])
        if self._add_goal_noise:
            goal += np.random.normal(scale=0.01, size=goal.shape)
        return goal

    def get_expert_goals(self):
        # assert False
        # goals = np.zeros((10, 10))
        # goals[:, 3:] = np.array([0, 0, -5.9625151e-04, -3.4385541e-04, 4.1548879e-04, 1.5108634e-04, 2.9286076e-07])
        # goals[:, :3] = self._sample_goal()
        # return goals
        return None

class FetchPushEnvSameLocReset(FetchPushEnv):
    def __init__(self, *args, camera='camera2', **kwargs):
        self._camera_name = camera
        super(FetchPushEnvSameLocReset, self).__init__(*args, **kwargs)
        self.sim.model.geom_rgba[1:5] = 0  # Hide the lasers

    def reset(self):
        super(FetchPushEnvSameLocReset, self).reset()
        object_qpos = np.array([1.21847399e+00,
                                6.17473012e-01,
                                4.24702091e-01,
                                1.00000000e+00,
                                -1.92607042e-07,
                                2.96318526e-07,
                                -2.88376574e-16])
        self.sim.data.set_joint_qpos('object0:joint', object_qpos)

        for _ in range(10):
          super(FetchPushEnvSameLocReset, self).step(np.array([0.0, 0.0, 0.0, 0.0]))

        s = self._get_obs()
        obs = self.observation(s)
        return obs


    def _viewer_setup(self):
      super(FetchPushEnvSameLocReset, self)._viewer_setup()
      if self._camera_name == 'camera1':
        self.viewer.cam.lookat[Ellipsis] = np.array([1.2, 0.8, 0.4])
        self.viewer.cam.distance = 0.9
        self.viewer.cam.azimuth = 180
        self.viewer.cam.elevation = -40
      elif self._camera_name == 'camera2':
        self.viewer.cam.lookat[Ellipsis] = np.array([1.25, 0.8, 0.4])
        self.viewer.cam.distance = 0.65
        self.viewer.cam.azimuth = 90
        self.viewer.cam.elevation = -40
      elif self._camera_name == 'camera3':
        self.viewer.cam.lookat[Ellipsis] = np.array([1.25, 0.8, 0.4])
        self.viewer.cam.distance = 0.9
        self.viewer.cam.azimuth = 90
        self.viewer.cam.elevation = -40
      else:
        raise NotImplementedError

class FetchPushEnvSameLocResetGoals(FetchPushEnvSameLocReset):
    def __init__(self, *args, add_goal_noise=False, **kwargs):
        self._add_goal_noise = add_goal_noise
        super(FetchPushEnvSameLocResetGoals, self).__init__(*args, **kwargs)

    def _sample_goal(self):
        goal = np.array([1.4, 0.9, 0.42469975])
        if self._add_goal_noise:
            goal += np.random.normal(scale=0.01, size=goal.shape)
        return goal

    def get_expert_goals(self):
        return None

class FetchReachImage(reach.FetchReachEnv):
  """Wrapper for the FetchReach environment with image observations."""

  def __init__(self):
    self._dist = []
    self._dist_vec = []
    super(FetchReachImage, self).__init__()
    self._old_observation_space = self.observation_space
    self._new_observation_space = gym.spaces.Box(
        low=np.full((64*64*6), 0),
        high=np.full((64*64*6), 255),
        dtype=np.uint8)
    self.observation_space = self._new_observation_space
    self.sim.model.geom_rgba[1:5] = 0  # Hide the lasers

  def reset_metrics(self):
    self._dist_vec = []
    self._dist = []

  def reset(self):
    if self._dist:  # if len(self._dist) > 0, ...
      self._dist_vec.append(self._dist)
    self._dist = []

    # generate the new goal image
    self.observation_space = self._old_observation_space
    s = super(FetchReachImage, self).reset()
    self.observation_space = self._new_observation_space
    self._goal = s['desired_goal'].copy()

    for _ in range(10):
      hand = s['achieved_goal']
      obj = s['desired_goal']
      delta = obj - hand
      a = np.concatenate([np.clip(10 * delta, -1, 1), [0.0]])
      s, _, _, _ = super(FetchReachImage, self).step(a)

    self._goal_img = self.observation(s)

    self.observation_space = self._old_observation_space
    s = super(FetchReachImage, self).reset()
    self.observation_space = self._new_observation_space
    img = self.observation(s)
    dist = np.linalg.norm(s['achieved_goal'] - self._goal)
    self._dist.append(dist)
    return np.concatenate([img, self._goal_img])

  def step(self, action):
    s, _, _, _ = super(FetchReachImage, self).step(action)
    dist = np.linalg.norm(s['achieved_goal'] - self._goal)
    self._dist.append(dist)
    done = False
    r = float(dist < 0.05)
    info = {}
    # info = dict(state=s)
    img = self.observation(s)
    return np.concatenate([img, self._goal_img]), r, done, info

  def observation(self, observation):
    self.sim.data.site_xpos[0] = 1_000_000
    img = self.render(mode='rgb_array', height=64, width=64)
    return img.flatten()

  def _viewer_setup(self):
    super(FetchReachImage, self)._viewer_setup()
    # self.viewer.cam.lookat[Ellipsis] = np.array([1.2, 0.8, 0.5])
    # self.viewer.cam.distance = 0.8
    # self.viewer.cam.azimuth = 180
    # self.viewer.cam.elevation = -30

    self.viewer.cam.lookat[Ellipsis] = np.array([1.2, 0.8, 0.5])
    self.viewer.cam.distance = 1.4
    self.viewer.cam.azimuth = 120
    self.viewer.cam.elevation = -30

class FetchReachImageGoals(FetchReachImage):
  def __init__(self, add_goal_noise=False):
      self._add_goal_noise = add_goal_noise
      super(FetchReachImageGoals, self).__init__()

  def _sample_goal(self): ###===### ###---###
      # return np.array([1.4, 0.8, 0.6])
      goal =  np.array([1.3, 0.3, 0.9]) #, 0, 0, -5.9625151e-04, -3.4385541e-04, 4.1548879e-04, 1.5108634e-04, 2.9286076e-07])
      if self._add_goal_noise:
          goal += np.random.normal(scale=0.01, size=goal.shape)
      return goal

  def get_expert_goals(self):
      return None

class FetchPushImage(push.FetchPushEnv):
  """Wrapper for the FetchPush environment with image observations."""

  def __init__(self, camera='camera2', start_at_obj=True, rand_y=False, same_block_start_pos=False):
    if same_block_start_pos:
        assert not rand_y
    self._start_at_obj = start_at_obj
    self._rand_y = rand_y
    self._same_block_start_pos = same_block_start_pos
    self._camera_name = camera
    self._dist = []
    self._dist_vec = []
    super(FetchPushImage, self).__init__()
    self._old_observation_space = self.observation_space
    self._new_observation_space = gym.spaces.Box(
        low=np.full((64*64*6), 0),
        high=np.full((64*64*6), 255),
        dtype=np.uint8)
    self.observation_space = self._new_observation_space
    self.sim.model.geom_rgba[1:5] = 0  # Hide the lasers

    print("self._same_block_start_pos:", self._same_block_start_pos)
    print("self._rand_y:", self._rand_y)

  def reset_metrics(self):
    self._dist_vec = []
    self._dist = []

  def _move_hand_to_obj(self):
    s = super(FetchPushImage, self)._get_obs()
    for _ in range(100):
      hand = s['observation'][:3]
      obj = s['achieved_goal'] + np.array([-0.02, 0.0, 0.0])
      delta = obj - hand
      if np.linalg.norm(delta) < 0.06:
        break
      a = np.concatenate([np.clip(delta, -1, 1), [0.0]])
      s, _, _, _ = super(FetchPushImage, self).step(a)

  def reset(self):
    if self._dist:  # if len(self._dist) > 0 ...
      self._dist_vec.append(self._dist)
    self._dist = []

    # generate the new goal image
    self.observation_space = self._old_observation_space
    s = super(FetchPushImage, self).reset()
    self.observation_space = self._new_observation_space

    if not self._same_block_start_pos:
        # Randomize object position
        for _ in range(8):
          super(FetchPushImage, self).step(np.array([-1.0, 0.0, 0.0, 0.0]))
    object_qpos = self.sim.data.get_joint_qpos('object0:joint')
    if not self._rand_y:
      object_qpos[1] = 0.75
    self.sim.data.set_joint_qpos('object0:joint', object_qpos)
    self._move_hand_to_obj()
    self._goal_img = self.observation(s)
    block_xyz = self.sim.data.get_joint_qpos('object0:joint')[:3]
    if block_xyz[2] < 0.4:  # If block has fallen off the table, recurse.
      print('Bad reset, recursing.')
      return self.reset()
    self._goal = block_xyz[:2].copy()

    self.observation_space = self._old_observation_space
    s = super(FetchPushImage, self).reset()
    self.observation_space = self._new_observation_space
    for _ in range(8):
      super(FetchPushImage, self).step(np.array([-1.0, 0.0, 0.0, 0.0]))
    object_qpos = self.sim.data.get_joint_qpos('object0:joint')
    object_qpos[:2] = np.array([1.15, 0.75])
    self.sim.data.set_joint_qpos('object0:joint', object_qpos)
    if self._start_at_obj:
      self._move_hand_to_obj()
    else:
      for _ in range(5):
        super(FetchPushImage, self).step(self.action_space.sample())

    block_xyz = self.sim.data.get_joint_qpos('object0:joint')[:3].copy()
    img = self.observation(s)
    dist = np.linalg.norm(block_xyz[:2] - self._goal)
    self._dist.append(dist)
    if block_xyz[2] < 0.4:  # If block has fallen off the table, recurse.
      print('Bad reset, recursing.')
      return self.reset()

    return np.concatenate([img, self._goal_img])

  def step(self, action):
    s, _, _, _ = super(FetchPushImage, self).step(action)
    block_xy = self.sim.data.get_joint_qpos('object0:joint')[:2]
    dist = np.linalg.norm(block_xy - self._goal)
    self._dist.append(dist)
    done = False
    r = float(dist < 0.05)  # Taken from the original task code.
    info = {}
    img = self.observation(s)
    return np.concatenate([img, self._goal_img]), r, done, info

  def observation(self, observation):
    self.sim.data.site_xpos[0] = 1_000_000
    img = self.render(mode='rgb_array', height=64, width=64)
    return img.flatten()


  def _viewer_setup(self):
    super(FetchPushImage, self)._viewer_setup()
    if self._camera_name == 'camera1':
      self.viewer.cam.lookat[Ellipsis] = np.array([1.2, 0.8, 0.4])
      self.viewer.cam.distance = 0.9
      self.viewer.cam.azimuth = 180
      self.viewer.cam.elevation = -40
    elif self._camera_name == 'camera2':
      self.viewer.cam.lookat[Ellipsis] = np.array([1.25, 0.8, 0.4])
      self.viewer.cam.distance = 0.65
      self.viewer.cam.azimuth = 90
      self.viewer.cam.elevation = -40
    elif self._camera_name == 'camera3':
      self.viewer.cam.lookat[Ellipsis] = np.array([1.25, 0.8, 0.4])
      self.viewer.cam.distance = 0.9
      self.viewer.cam.azimuth = 90
      self.viewer.cam.elevation = -40
    else:
      raise NotImplementedError

class FetchPushImageGoals(FetchPushImage):
    def __init__(self, *args, add_goal_noise=False, **kwargs):
        self._add_goal_noise = add_goal_noise
        super(FetchPushImageGoals, self).__init__(*args, **kwargs)

    def _sample_goal(self):
        goal = np.array([1.4, 0.9, 0.42469975])
        if self._add_goal_noise:
            goal += np.random.normal(scale=0.01, size=goal.shape)
        return goal

    def get_expert_goals(self):
        return None

class FetchPushEnv3(push.FetchPushEnv):
  """Wrapper for the FetchPush environment with image observations."""

  def __init__(self, camera='camera2', start_at_obj=True, rand_y=False, same_block_start_pos=False, use_natural_goal=False):
    if same_block_start_pos:
        assert not rand_y
    self._start_at_obj = start_at_obj
    self._rand_y = rand_y
    self._same_block_start_pos = same_block_start_pos
    self._camera_name = camera
    self._dist = []
    self._dist_vec = []
    super(FetchPushEnv3, self).__init__()
    # self._old_observation_space = self.observation_space
    # self._new_observation_space = gym.spaces.Box(
    #     low=np.full((64*64*6), 0),
    #     high=np.full((64*64*6), 255),
    #     dtype=np.uint8)
    # self.observation_space = self._new_observation_space

    self._old_observation_space = self.observation_space
    self._new_observation_space = gym.spaces.Box(
        low=np.full((50,), -np.inf),
        high=np.full((50,), np.inf),
        dtype=np.float32)
    self.observation_space = self._new_observation_space

    self.sim.model.geom_rgba[1:5] = 0  # Hide the lasers

    print("self._same_block_start_pos:", self._same_block_start_pos)
    print("self._rand_y:", self._rand_y)

    self._use_natural_goal = use_natural_goal

    print("self._use_natural_goal:", self._use_natural_goal)

  def reset_metrics(self):
    self._dist_vec = []
    self._dist = []

  def _move_hand_to_obj(self):
    s = super(FetchPushEnv3, self)._get_obs()
    for _ in range(100):
      hand = s['observation'][:3]
      obj = s['achieved_goal'] + np.array([-0.02, 0.0, 0.0])
      delta = obj - hand
      if np.linalg.norm(delta) < 0.06:
        break
      a = np.concatenate([np.clip(delta, -1, 1), [0.0]])
      s, _, _, _ = super(FetchPushEnv3, self).step(a)

  def reset(self):
    if self._dist:  # if len(self._dist) > 0 ...
      self._dist_vec.append(self._dist)
    self._dist = []

    # generate the new goal image
    self.observation_space = self._old_observation_space
    s = super(FetchPushEnv3, self).reset()
    self.observation_space = self._new_observation_space

    if not self._same_block_start_pos:
        # Randomize object position
        for _ in range(8):
          super(FetchPushEnv3, self).step(np.array([-1.0, 0.0, 0.0, 0.0]))
    object_qpos = self.sim.data.get_joint_qpos('object0:joint')
    if not self._rand_y:
      object_qpos[1] = 0.75
    self.sim.data.set_joint_qpos('object0:joint', object_qpos)
    self._move_hand_to_obj()
    # self._goal_img = self.observation(s)
    block_xyz = self.sim.data.get_joint_qpos('object0:joint')[:3]
    if block_xyz[2] < 0.4:  # If block has fallen off the table, recurse.
      print('Bad reset, recursing.')
      return self.reset()
    self._goal = block_xyz[:2].copy()

    self.observation_space = self._old_observation_space
    s = super(FetchPushEnv3, self).reset()
    self.observation_space = self._new_observation_space
    for _ in range(8):
      super(FetchPushEnv3, self).step(np.array([-1.0, 0.0, 0.0, 0.0]))
    object_qpos = self.sim.data.get_joint_qpos('object0:joint')
    object_qpos[:2] = np.array([1.15, 0.75])
    self.sim.data.set_joint_qpos('object0:joint', object_qpos)
    if self._start_at_obj:
      self._move_hand_to_obj()
    else:
      for _ in range(5):
        super(FetchPushEnv3, self).step(self.action_space.sample())

    block_xyz = self.sim.data.get_joint_qpos('object0:joint')[:3].copy()
    # img = self.observation(s)
    dist = np.linalg.norm(block_xyz[:2] - self._goal)
    self._dist.append(dist)
    if block_xyz[2] < 0.4:  # If block has fallen off the table, recurse.
      print('Bad reset, recursing.')
      return self.reset()

    # return np.concatenate([img, self._goal_img])
    return self.state_observation(s)

  def step(self, action):
    s, _, _, _ = super(FetchPushEnv3, self).step(action)
    block_xy = self.sim.data.get_joint_qpos('object0:joint')[:2]
    dist = np.linalg.norm(block_xy - self._goal)
    self._dist.append(dist)
    done = False
    r = float(dist < 0.05)  # Taken from the original task code.
    info = {}
    # img = self.observation(s)
    # return np.concatenate([img, self._goal_img]), r, done, info
    return self.state_observation(s), r, done, info

  def observation(self, observation):
    # self.sim.data.site_xpos[0] = 1_000_000
    # img = self.render(mode='rgb_array', height=64, width=64)
    # return img.flatten()
    raise NotImplementedError

  def _viewer_setup(self):
    super(FetchPushEnv3, self)._viewer_setup()
    if self._camera_name == 'camera1':
      self.viewer.cam.lookat[Ellipsis] = np.array([1.2, 0.8, 0.4])
      self.viewer.cam.distance = 0.9
      self.viewer.cam.azimuth = 180
      self.viewer.cam.elevation = -40
    elif self._camera_name == 'camera2':
      self.viewer.cam.lookat[Ellipsis] = np.array([1.25, 0.8, 0.4])
      self.viewer.cam.distance = 0.65
      self.viewer.cam.azimuth = 90
      self.viewer.cam.elevation = -40
    elif self._camera_name == 'camera3':
      self.viewer.cam.lookat[Ellipsis] = np.array([1.25, 0.8, 0.4])
      self.viewer.cam.distance = 0.9
      self.viewer.cam.azimuth = 90
      self.viewer.cam.elevation = -40
    else:
      raise NotImplementedError

  def state_observation(self, observation):
    start_index = 3
    end_index = 6
    goal_pos_1 = observation['achieved_goal']
    goal_pos_2 = observation['observation'][start_index:end_index]
    assert np.all(goal_pos_1 == goal_pos_2)
    s = observation['observation']
    g = np.zeros_like(s)

    if self._use_natural_goal:
        g = np.array([1.34207559e+00,  9.35664892e-01,  4.19158936e-01,  1.38074136e+00,
                  9.01960373e-01,  4.24771219e-01,  3.86658683e-02, -3.37044820e-02,
                  5.61230257e-03,  0.00000000e+00,  0.00000000e+00,  8.61858207e-05,
                 -2.82435009e-04, -1.29339844e-01, -6.43660547e-04, -2.83353089e-04,
                  1.36204704e-04, -4.84256343e-05,  1.58692084e-04,  1.01813814e-13,
                  6.47627865e-04,  2.84563721e-04, -1.28721018e-04,  8.95160338e-06,
                  2.86602793e-04])

    g[:start_index] = observation['desired_goal']
    g[start_index:end_index] = observation['desired_goal']



    # g[3:] = np.array([0, 0, -5.9625151e-04, -3.4385541e-04, 4.1548879e-04, 1.5108634e-04, 2.9286076e-07])

    return np.concatenate([s, g]).astype(np.float32)

class FetchPushEnv3Goals(FetchPushEnv3):
    def __init__(self, *args, add_goal_noise=False, **kwargs):
        self._add_goal_noise = add_goal_noise
        super(FetchPushEnv3Goals, self).__init__(*args, **kwargs)

    def _sample_goal(self):
        goal = np.array([1.4, 0.9, 0.42469975])
        if self._add_goal_noise:
            goal += np.random.normal(scale=0.01, size=goal.shape)
        return goal

    def get_expert_goals(self):
        return None

class FetchPushImageMinimalGoalsColors(FetchPushImageMinimalGoals):
    def reset(self):

        self.sim.model.geom_rgba[23, :3] = np.random.uniform(0, 1, 3)
        return super(FetchPushImageMinimalGoalsColors, self).reset()

class FetchPushImageMinimalGoalsOccluded(FetchPushImageMinimalGoals):
    def _viewer_setup(self):
      super(FetchPushImageMinimalGoalsOccluded, self)._viewer_setup()
      self.viewer.cam.lookat[Ellipsis] = np.array([1.25, 0.8, 0.4])
      self.viewer.cam.distance = 0.7
      self.viewer.cam.azimuth = 180.0
      self.viewer.cam.elevation = -80.0




"""
class FetchPushImageMinimal(push.FetchPushEnv):
     def __init__(self):
        super(FetchPushImageMinimal, self).__init__()
        # self._old_observation_space = self.observation_space
        # self._new_observation_space = gym.spaces.Box(
        #     low=np.full((50,), -np.inf),
        #     high=np.full((50,), np.inf),
        #     dtype=np.float32)
        # self.observation_space = self._new_observation_space

        self._old_observation_space = self.observation_space
        self._new_observation_space = gym.spaces.Box(
            low=np.full((64*64*6), 0),
            high=np.full((64*64*6), 255),
            dtype=np.uint8)
        self.observation_space = self._new_observation_space
        self.sim.model.geom_rgba[1:5] = 0  # Hide the lasers

    def reset_metrics(self):
      self._dist_vec = []
      self._dist = []

     def reset(self):
        if self._dist:  # if len(self._dist) > 0 ...
          self._dist_vec.append(self._dist)
        self._dist = []

        self.observation_space = self._old_observation_space
        s = super(FetchPushImageMinimal, self).reset()
        self.observation_space = self._new_observation_space

        self._goal_img = self.observation(s)

        block_xyz = self.sim.data.get_joint_qpos('object0:joint')[:3].copy()
        img = self.observation(s)
        dist = np.linalg.norm(block_xyz[:2] - self._goal)
        self._dist.append(dist)
        if block_xyz[2] < 0.4:  # If block has fallen off the table, recurse.
          print('Bad reset, recursing.')
          return self.reset()

        return np.concatenate([img, self._goal_img])



     def step(self, action):
        s, _, _, _ = super(FetchPushImageMinimal, self).step(action)
        done = False
        dist = np.linalg.norm(s['achieved_goal'] - s['desired_goal'])
        r = float(dist < 0.05)
        info = {}
        # info = dict(state=self.observation(s))
        return self.observation(s), r, done, info

     def observation(self, observation):
        start_index = 3
        end_index = 6
        goal_pos_1 = observation['achieved_goal']
        goal_pos_2 = observation['observation'][start_index:end_index]
        assert np.all(goal_pos_1 == goal_pos_2)
        s = observation['observation']
        g = np.zeros_like(s)
        g[:start_index] = observation['desired_goal']
        g[start_index:end_index] = observation['desired_goal']
        return np.concatenate([s, g]).astype(np.float32)

class FetchPushEnv2(push.FetchPushEnv):
  Wrapper for the FetchPush environment with image observations.

  def __init__(self, camera='camera2', start_at_obj=True, rand_y=False, same_block_start_pos=False):
    if same_block_start_pos:
        assert not rand_y
    self._start_at_obj = start_at_obj
    self._rand_y = rand_y
    self._same_block_start_pos = same_block_start_pos
    self._camera_name = camera
    self._dist = []
    self._dist_vec = []
    super(FetchPushEnv2, self).__init__()
    # self._old_observation_space = self.observation_space
    # self._new_observation_space = gym.spaces.Box(
    #     low=np.full((64*64*6), 0),
    #     high=np.full((64*64*6), 255),
    #     dtype=np.uint8)
    # self.observation_space = self._new_observation_space
    self._old_observation_space = self.observation_space
    self._new_observation_space = gym.spaces.Box(
        low=np.full((50,), -np.inf),
        high=np.full((50,), np.inf),
        dtype=np.float32)
    self.observation_space = self._new_observation_space
    self.sim.model.geom_rgba[1:5] = 0  # Hide the lasers

    print("self._same_block_start_pos:", self._same_block_start_pos)
    print("self._rand_y:", self._rand_y)

  def reset_metrics(self):
    self._dist_vec = []
    self._dist = []

  def _move_hand_to_obj(self):
    s = super(FetchPushEnv2, self)._get_obs()
    for _ in range(100):
      hand = s['observation'][:3]
      obj = s['achieved_goal'] + np.array([-0.02, 0.0, 0.0])
      delta = obj - hand
      if np.linalg.norm(delta) < 0.06:
        break
      a = np.concatenate([np.clip(delta, -1, 1), [0.0]])
      s, _, _, _ = super(FetchPushEnv2, self).step(a)

  def reset(self):
    if self._dist:  # if len(self._dist) > 0 ...
      self._dist_vec.append(self._dist)
    self._dist = []

    # generate the new goal image
    self.observation_space = self._old_observation_space
    s = super(FetchPushEnv2, self).reset()
    self.observation_space = self._new_observation_space

    if not self._same_block_start_pos:
        # Randomize object position
        for _ in range(8):
          super(FetchPushEnv2, self).step(np.array([-1.0, 0.0, 0.0, 0.0]))
    object_qpos = self.sim.data.get_joint_qpos('object0:joint')
    if not self._rand_y:
      object_qpos[1] = 0.75
    self.sim.data.set_joint_qpos('object0:joint', object_qpos)
    self._move_hand_to_obj()
    # self._goal_img = self.observation(s)
    block_xyz = self.sim.data.get_joint_qpos('object0:joint')[:3]
    if block_xyz[2] < 0.4:  # If block has fallen off the table, recurse.
      print('Bad reset, recursing.')
      return self.reset()
    self._goal = block_xyz[:2].copy()

    self.observation_space = self._old_observation_space
    s = super(FetchPushEnv2, self).reset()
    self.observation_space = self._new_observation_space
    for _ in range(8):
      super(FetchPushEnv2, self).step(np.array([-1.0, 0.0, 0.0, 0.0]))
    object_qpos = self.sim.data.get_joint_qpos('object0:joint')
    object_qpos[:2] = np.array([1.15, 0.75])
    self.sim.data.set_joint_qpos('object0:joint', object_qpos)
    if self._start_at_obj:
      self._move_hand_to_obj()
    else:
      for _ in range(5):
        super(FetchPushEnv2, self).step(self.action_space.sample())

    block_xyz = self.sim.data.get_joint_qpos('object0:joint')[:3].copy()
    # img = self.observation(s)
    dist = np.linalg.norm(block_xyz[:2] - self._goal)
    self._dist.append(dist)
    if block_xyz[2] < 0.4:  # If block has fallen off the table, recurse.
      print('Bad reset, recursing.')
      return self.reset()

    # return np.concatenate([img, self._goal_img])
    return self.observation(s)

  def step(self, action):
    s, _, _, _ = super(FetchPushEnv2, self).step(action)
    block_xy = self.sim.data.get_joint_qpos('object0:joint')[:2]
    dist = np.linalg.norm(block_xy - self._goal)
    self._dist.append(dist)
    done = False
    r = float(dist < 0.05)  # Taken from the original task code.
    info = {}
    # img = self.observation(s)
    # return np.concatenate([img, self._goal_img]), r, done, info
    return self.observation(s), r, done, info

  def observation(self, observation):
    # self.sim.data.site_xpos[0] = 1_000_000
    # img = self.render(mode='rgb_array', height=64, width=64)
    # return img.flatten()
    start_index = 3
    end_index = 6
    goal_pos_1 = observation['achieved_goal']
    goal_pos_2 = observation['observation'][start_index:end_index]
    assert np.all(goal_pos_1 == goal_pos_2)
    s = observation['observation']
    g = np.zeros_like(s)
    g[:start_index] = observation['desired_goal']
    g[start_index:end_index] = observation['desired_goal']
    return np.concatenate([s, g]).astype(np.float32)

  def _viewer_setup(self):
    super(FetchPushEnv2, self)._viewer_setup()
    if self._camera_name == 'camera1':
      self.viewer.cam.lookat[Ellipsis] = np.array([1.2, 0.8, 0.4])
      self.viewer.cam.distance = 0.9
      self.viewer.cam.azimuth = 180
      self.viewer.cam.elevation = -40
    elif self._camera_name == 'camera2':
      self.viewer.cam.lookat[Ellipsis] = np.array([1.25, 0.8, 0.4])
      self.viewer.cam.distance = 0.65
      self.viewer.cam.azimuth = 90
      self.viewer.cam.elevation = -40
    elif self._camera_name == 'camera3':
      self.viewer.cam.lookat[Ellipsis] = np.array([1.25, 0.8, 0.4])
      self.viewer.cam.distance = 0.9
      self.viewer.cam.azimuth = 90
      self.viewer.cam.elevation = -40
    else:
      raise NotImplementedError

class FetchPushEnv2Goals(FetchPushEnv2):
        def __init__(self, *args, add_goal_noise=False, **kwargs):
            self._add_goal_noise = add_goal_noise
            super(FetchPushEnv2Goals, self).__init__(*args, **kwargs)

        def _sample_goal(self):
            goal = np.array([1.4, 0.9, 0.42469975])
            if self._add_goal_noise:
                goal += np.random.normal(scale=0.01, size=goal.shape)
            return goal

        def get_expert_goals(self):
            return None

"""
