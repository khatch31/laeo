import os
import numpy as np
import gym
import atexit
import functools
import sys
import threading
import traceback
import mujoco_py
from PIL import Image
import threading
from dm_control.mujoco import engine

OBS_ELEMENT_INDICES = {
    'bottom burner': np.array([11, 12]),
    'top burner': np.array([15, 16]),
    'light switch': np.array([17, 18]),
    'slide cabinet': np.array([19]),
    'hinge cabinet': np.array([20, 21]),
    'microwave': np.array([22]),
    'kettle': np.array([23, 24, 25, 26, 27, 28, 29]),
    }

OBS_ELEMENT_GOALS = {
    'bottom burner': np.array([-0.88, -0.01]),
    'top burner': np.array([-0.92, -0.01]),
    'light switch': np.array([-0.69, -0.05]),
    'slide cabinet': np.array([0.37]),
    'hinge cabinet': np.array([0., 1.45]),
    'microwave': np.array([-0.75]),
    'kettle': np.array([-0.23, 0.75, 1.62, 0.99, 0., 0., -0.06]),
    }

BONUS_THRESH = 0.3

class Kitchen:
    def __init__(self, task=['microwave'], size=(64, 64), initial_states=None):
        # from .RPL.adept_envs import adept_envs
        sys.path.append("/iris/u/khatch/preliminary_experiments/model_based_offline_online/relay-policy-learning/adept_envs")
        import adept_envs
        # self._env = gym.make('kitchen_relax-v1')
        self._env = gym.make('kitchen_relax_rpl-v1')
        self._task = task
        self._img_h = size[0]
        self._img_w = size[1]
        self._initial_states = initial_states
        self.tasks_to_complete = ['bottom burner',
                                  'top burner',
                                  'light switch',
                                  'slide cabinet',
                                  'hinge cabinet',
                                  'microwave',
                                  'kettle']

        # The original RPL env already includes a goal obs in the observation
        # which is all zeros

        # self._old_observation_space = self.observation_space
        # self._new_observation_space = gym.spaces.Box(
        #     low=np.full((120,), -np.inf),
        #     high=np.full((120,), np.inf),
        #     dtype=np.float32)
        # self.observation_space = self._new_observation_space

    def __getattr__(self, attr):
        if attr == '_wrapped_env':
            raise AttributeError()
        return getattr(self._env, attr)

    def step(self, *args, **kwargs):
        obs, reward, done, info = self._env.step(*args, **kwargs)
        reward_dict = self._compute_reward_dict(obs)

        # if self._image:
        #     img = self.render(mode='rgb_array', size=(self._img_h, self._img_w))
        #     obs_dict = dict(image=img)
        #
        #     if self._proprio:
        #         obs_dict["proprio"] = obs[:9]
        # else:
        #     obs = np.concatenate((obs, np.zeros_like(obs)), axis=0)
        # obs = np.concatenate((obs, np.zeros_like(obs)), axis=0)


        reward = sum([reward_dict[obj] for obj in self._task])

        # obs_dict.update({"reward " + key:float(val) for key, val in reward_dict.items()})
        # return obs_dict, reward, done, info
        return obs, reward, done, info

    def reset(self, *args, **kwargs):
        # self.observation_space = self._old_observation_space
        obs = self._env.reset(*args, **kwargs)
        # self.observation_space = self._new_observation_space

        # print("obs:", obs)

        if self._initial_states is not None:
            idxs = np.arange(self._initial_states["qpos"].shape[0])
            idx = np.random.choice(idxs)

            self._env.sim.data.qpos[:] = self._initial_states["qpos"][idx].copy()
            self._env.sim.data.qvel[:] = self._initial_states["qvel"][idx].copy()
            self._env.sim.forward()

            obs = self._env.get_obs()

            # Make a jupyternotebook to test the reset method

        # if self._image:
        #     img = self.render(mode='rgb_array', size=(self._img_h, self._img_w))
        #     obs_dict = dict(image=img)
        #
        #     if self._proprio:
        #         obs_dict["proprio"] = obs[:9]
        #     return obs_dict
        # else:
        #     obs = np.concatenate((obs, np.zeros_like(obs)), axis=0)
        #     return obs

        # print("obs:", obs)
        # obs = np.concatenate((obs, np.zeros_like(obs)), axis=0)
        # print("obs:", obs)
        return obs

    def _compute_reward_dict(self, obs):
        reward_dict = {}
        for element in self.tasks_to_complete:
            element_idx = OBS_ELEMENT_INDICES[element]
            obs_obj = obs[..., element_idx]
            obs_goal = OBS_ELEMENT_GOALS[element]
            distance = np.linalg.norm(obs_obj - obs_goal)
            complete = distance < BONUS_THRESH
            reward_dict[element] = complete

            obs_dict = self.obs_dict
        return reward_dict

    def render(self, mode='human', size=(1920, 2550)):
        if mode =='rgb_array':
            # camera = engine.MovableCamera(self.sim, 1920, 2560)
            camera = engine.MovableCamera(self._env.sim, size[0], size[1])
            # camera.set_pose(distance=2.2, lookat=[-0.2, .5, 2.], azimuth=70, elevation=-35)
            camera.set_pose(distance=1.86, lookat=[-0.3, .5, 2.], azimuth=90, elevation=-60)
            img = camera.render()
            return img
        else:
            # super(KitchenTaskRelaxV1, self).render()
            self._env.render(mode, size)

    def get_expert_goals(self):
        return None

    # @property
    # def observation_space(self):
    #     spaces = {}
    #     spaces['image'] = gym.spaces.Box(0, 255, (self._img_h, self._img_w, 3), dtype=np.uint8)
    #
    #     if self._proprio:
    #         spaces["proprio"] = gym.spaces.Box(-np.inf, np.inf, (9,), dtype=np.float32)
    #
    #     return gym.spaces.Dict(spaces)


class KitchenImage:
    pass
