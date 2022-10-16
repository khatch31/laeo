import inspect
import os

from absl import logging
import d4rl  # pylint: disable=unused-import
# import gin
import gym

# import sys
# # sys.path.append("/iris/u/khatch/anaconda3/envs/combo/lib/python3.7/site-packages")
# sys.path.append("/iris/u/khatch/metaworld_combo/")
# from metaworld_combo.envs.mujoco.sawyer_xyz.v2.sawyer_drawer_open_v2 import SawyerDrawerOpenEnvV2
# from metaworld_combo.envs.mujoco.sawyer_xyz.v2.sawyer_door_v2 import SawyerDoorEnvV2
#
# sys.path.append("/iris/u/khatch/metaworld_door")
# # sys.path.insert(0, "/iris/u/khatch/metaworld_door")
# import inspect
# import metaworld
# # from importlib.machinery import SourceFileLoader
# # metaworld = SourceFileLoader("metaworld", "/iris/u/khatch/metaworld_door/metaworld/__init__.py").load_module()
# print("inspect.getfile(metaworld):", inspect.getfile(metaworld))
#
# from metaworld.envs.mujoco.sawyer_xyz.v2 import SawyerMultitaskDoorEnvV2, SawyerMultitaskDoorCloseEnvV2, SawyerMultitaskDrawerOpenEnvV2, SawyerMultitaskDrawerCloseEnvV2
# from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_multimodal_envs import SawyerMultimodalDrawerOpenEnvV2, SawyerMultimodalDoorOpenEnvV2, SawyerMultimodalLeverPullEnvV2, SawyerMultimodalPlateSlideEnvV2, SawyerDialTurnEnvV2
#
# from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_drawer_door_env import SawyerDrawerDoorDrawerOpenEnvV2, SawyerDrawerDoorDoorOpenEnvV2


from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_dial_turn_v2 import SawyerDialTurnEnvV2


import mujoco_py
from dm_control import mujoco
from collections import defaultdict

# Modified files to get metaworld_combo to work
# /iris/u/khatch/metaworld_combo/metaworld_combo/envs/mujoco/sawyer_xyz/v2/__init__.py
# /iris/u/khatch/metaworld_combo/metaworld_combo/envs/mujoco/sawyer_xyz/sawyer_xyz_env.py
# /iris/u/khatch/metaworld_combo/metaworld_combo/envs/mujoco/sawyer_xyz/v2/sawyer_drawer_open_v2.py
# /iris/u/khatch/metaworld_combo/metaworld_combo/envs/mujoco/env_dict.py
# /iris/u/khatch/metaworld/metaworld_combo/__init__.py



# import robodesk
import numpy as np
# from tf_agents.environments import suite_gym
# from tf_agents.environments import tf_py_environment
import tqdm
# from wrappers import ActionRepeatEnv, BinaryRewardEnv, ImageEnv, BlankInfoWrapper
import cv2

# from mujoco_py import GlfwContext
# GlfwContext(offscreen=True)  # Create a window to init GLFW.


# We need to import d4rl so that gym registers the environments.
os.environ['SDL_VIDEODRIVER'] = 'dummy'

import numpy as np

from collections import defaultdict
# from tf_agents.environments.wrappers import PyEnvironmentBaseWrapper
from gym.spaces import Box, Dict

def get_env(task_name, image_obs=False):
    # default_action_repeat = 4
    # task_name = env_name.split(":")[-1]
    # elif "sawyer:multimodal" in env_name or "sawyer:drawer_door" in env_name:




    # debug_env = SawyerDialTurnEnvV2()
    # viewer = mujoco_py.MjRenderContextOffscreen(debug_env.sim, device_id=-1)
    # image = debug_env.render("rgb_array")
    # print("debug_env:", debug_env)
    # print("viewer:", viewer)
    # print("image.shape:", image.shape)


    gym_env = SawyerMultimodalEnv(task_name, size=(64, 64))
    # default_max_episode_steps = 50
    info_aggr_fns = {"success":lambda x:float(min(np.sum(x), 1))}
    for key in gym_env.base_env.info_keys:
        if "state/" in key:
            info_aggr_fns[key] = lambda x:x[-1]

    gym_env = ActionRepeatEnv(gym_env, 4, info_aggr_fns)

    if image_obs:
        gym_env = ImageEnv(gym_env)

    gym_env = BlankGoalEnv(gym_env)
    gym_env = BinaryRewardEnv(gym_env)

    return gym_env, gym_env.observation_space.shape[0] // 2


class SawyerCOMBOBase:
    def __init__(self, size=(128, 128)):
        self._build_env()
        self._setup_viewer()
        self.size = size

    def __getattr__(self, attr):
        if attr == '_wrapped_env':
            raise AttributeError()
        return getattr(self._env, attr)

    @property
    def base_env(self):
        return self

    @property
    def observation_space(self):
        return self._env.observation_space

    def step(self, action):
        state, reward, done, info = self._env.step(action)

        for key, val in info.items():
            if val is None:
                info[key] = 0

        return state, reward, done, info

    def reset(self, **kwargs):
        state = self._env.reset()
        state = self._env.reset()
        return state

    # def get_image(self, width=84, height=84, camera_name=None):
    #     return self.sim.render(
    #         width=width,
    #         height=height,
    #         camera_name=camera_name,
    #     )

    # def render(self, mode='rgb_array', width = 128, height = 128):
    def render(self, mode='rgb_array', **kwargs):
        # self.viewer.render(width=width, height=width)
        # self.viewer.render(width=self.size[0], height=self.size[1])
        img = self.viewer.read_pixels(self.size[0], self.size[1], depth=False)
        img = img[::-1]
        return img

    def _get_blank_info(self):
        raise NotImplementedError

    def get_dataset(self, num_obs=256):
        raise NotImplementedError

    def _get_expert_obs(self, *args, **kwargs):
        raise NotImplementedError

    def _setup_viewer(self):
        raise NotImplementedError

    def _build_env(self):
        raise NotImplementedError

    def get_dataset(self, num_obs=256):
        raise NotImplementedError

class SawyerMultimodalEnv(SawyerCOMBOBase):
    def __init__(self, task, size=(128, 128)):
        self._build_env(task)
        self._setup_viewer()
        self.size = size

    def get_expert_goals(self):
        print("In get expert goals")
        return None

    def _build_env(self, task):
        if "drawer_door_drawer_open" in task:
            self._env = SawyerDrawerDoorDrawerOpenEnvV2()
        elif "drawer_door_door_open" in task:
            self._env = SawyerDrawerDoorDoorOpenEnvV2()
        elif "drawer_open" in task:
            self._env = SawyerMultimodalDrawerOpenEnvV2()
        elif "door_open" in task:
            self._env = SawyerMultimodalDoorOpenEnvV2()
        elif "lever_pull" in task:
            self._env = SawyerMultimodalLeverPullEnvV2()
        elif "plate_slide" in task:
            self._env = SawyerMultimodalPlateSlideEnvV2()
        elif "dial_turn" in task:
            self._env = SawyerDialTurnEnvV2()
        else:
            raise ValueError(f"Unsupported task: \"{task}\".")

    def _setup_viewer(self):
        #Setup camera in environment
        # self.viewer = mujoco_py.MjRenderContextOffscreen(self._env.sim, -1)
        # self.viewer = mujoco_py.MjRenderContextOffscreen(self._env.sim, 0)

        import pdb; pdb.set_trace()
        import inspect
        inspect.getfile(mujoco_py)
        inspect.getfile(metaworld)
        self.viewer = mujoco_py.MjRenderContextOffscreen(self._env.sim, device_id=-1)
        # # Original
        self.viewer.cam.elevation = -45
        self.viewer.cam.azimuth = 160
        self.viewer.cam.distance = 1.75
        self.viewer.cam.lookat[0] = 0.075
        self.viewer.cam.lookat[1] = 0.75
        self.viewer.cam.lookat[2] = 0.15

    def _get_blank_info(self):
        if self._env._target_pos is None:
            self._env.reset()

        action = np.zeros(self._env.action_space.shape)
        ob = self._get_obs()
        reward, reachDist, pullDist = self.compute_reward(action, ob)


        full_state = self.__getstate__()

        env_state = full_state["env_state"]
        joint_state, mocap_state = env_state
        mocap_pos, mocap_quat = mocap_state

        time, qpos, qvel, act, udd_state = joint_state


        info = {
            'reachDist': reachDist,
            'goalDist': pullDist,
            'epRew': reward,
            "state/time":time,
            "state/qpos":qpos,
            "state/qvel":qvel,
            "state/mocap_pos":mocap_pos,
            "state/mocap_quat":mocap_quat,
            # "full_state":full_state,
            'success': self._compute_success(reachDist, pullDist),}

        return info


class WrapperEnv:
    def __init__(self, env):
        self._env = env

    def reset(self, *args, **kwargs):
        return self._env.reset(*args, **kwargs)

    def step(self, *args, **kwargs):
        return self._env.step(*args, **kwargs)

    def render(self, *args, **kwargs):
        return self._env.render(*args, **kwargs)

    def get_dataset(self, *args, **kwargs):
        return self._env.get_dataset(*args, **kwargs)

    @property
    def base_env(self):
        return self._env.base_env

    @property
    def observation_space(self):
        return self._env.observation_space

    @property
    def action_space(self):
        return self._env.action_space

    def _get_blank_info(self):
        return self._env._get_blank_info()

    def get_expert_goals(self):
        return self._env.get_expert_goals()

class BlankGoalEnv(WrapperEnv):
    @property
    def observation_space(self):
        low = np.concatenate((self._env.observation_space.low, np.zeros_like(self._env.observation_space.low)))
        high = np.concatenate((self._env.observation_space.high, np.zeros_like(self._env.observation_space.high)))
        new_observation_space = gym.spaces.Box(
            low=low[:42], ###$$$###
            high=high[:42],
            dtype=np.float32)
        return new_observation_space

    def reset(self, *args, **kwargs):
        # obs = self._env.reset(*args, **kwargs)
        # obs = np.concatenate((obs, np.zeros_like(obs)), axis=0)
        # return obs
        ###$$$###
        return self._env.observation_space.sample()

    def step(self, *args, **kwargs):
        obs, reward, done, info = self._env.step(*args, **kwargs)
        obs = np.concatenate((obs, np.zeros_like(obs)), axis=0)
        return obs, reward, done, info

class ActionRepeatEnv(WrapperEnv):
    def __init__(self, env, amount, info_aggr_fns=None):
        super().__init__(env)
        self._amount = amount
        self._info_aggr_fns = info_aggr_fns or {}

    # def __getattr__(self, name):
    #     return getattr(self._env, name)

    def step(self, action):
        done = False
        total_reward = 0
        infos = defaultdict(list)
        current_step = 0
        while current_step < self._amount and not done:
            obs, reward, done, info = self._env.step(action)
            total_reward += reward
            for key, val in info.items():
                infos[key].append(val)
            current_step += 1
            # print("\treward: {}, epRew: {}".format(reward, info["epRew"]))

        total_info = {}#defaultdict(list)
        for key, val in infos.items():
            if key in self._info_aggr_fns:
                total_info[key] = self._info_aggr_fns[key](val)
            else:
                total_info[key] = np.sum(val, axis=0)
        # print("total_reward: {}, total epReward: {}".format(total_reward, total_info["epRew"]))
        return obs, total_reward, done, total_info

class BinaryRewardEnv(WrapperEnv):
    def step(self, *args, **kwargs):
        state, reward, done, info = self._env.step(*args, **kwargs)
        reward = 1.0 * info['success']
        return state, reward, done, info

class ObsDictEnv(WrapperEnv):
    def __init__(self, env, obs_key="observation"):
        super().__init__(env)
        self._obs_key = obs_key

    def reset(self, **kwargs):
        obs = self._env.reset(**kwargs)
        obs_dict = {self._obs_key:obs}
        return obs_dict

    def step(self, *args, **kwargs):
        obs, reward, done, info = self._env.step(*args, **kwargs)
        obs_dict = {self._obs_key:obs}
        return obs_dict, reward, done, info

    @property
    def observation_space(self):
        obs_space = self._env.observation_space
        obs_key = self._obs_key
        return Dict({self._obs_key:obs_space})


class ImageEnv(WrapperEnv):
    def __init__(self, env, channels_first=False):
        super().__init__(env)
        self._channels_first = channels_first

    def reset(self, **kwargs):
        state = self._env.reset()
        image = self._env.render().astype(np.float32)# / 255
        # obs[:, :, 0] = 0
        # obs[:, :, 1] = 0.5
        # obs[:, :, 2] = 1

        if self._channels_first:
            image = np.moveaxis(image, -1, 0)

        image = image.flatten()
        return image

    def step(self, *args, **kwargs):
        state, reward, done, info = self._env.step(*args, **kwargs)
        image = self._env.render().astype(np.float32)# / 255
        # obs[:, :, 0] = 0
        # obs[:, :, 1] = 0.5
        # obs[:, :, 2] = 1

        if self._channels_first:
            image = np.moveaxis(image, -1, 0)

        image = image.flatten()

        info["state"] = state
        return image, reward, done, info

    def _get_blank_info(self):
        info = self._env._get_blank_info()
        state = self.base_env._get_obs()
        info["state"] = state
        return info

    @property
    def observation_space(self):
        # return Box(0, 1, (self.base_env.size[0] * self.base_env.size[1] * 3,), dtype=np.float32)
        return Box(0, 1, (self.base_env.size[0] * self.base_env.size[1] * 3,), dtype=np.uint8)

class ProprioWrapper(WrapperEnv):
    def reset(self, **kwargs):
        state = self._env.reset(**kwargs)
        obs_dict = self._env.get_obs_dict()
        return np.concatenate((state, obs_dict['proprio']), axis=-1)

    def step(self, *args, **kwargs):
        state, reward, done, info = self._env.step(*args, **kwargs)
        state = np.concatenate((state, info['proprio']), axis=-1)
        return image, reward, done, info

    @property
    def observation_space(self):
        obs_space = self._env.observation_space

        obs_dict = self._env.get_obs_dict()
        obs_dict['proprio']

        return Box(0, 1, (obs_space.shape[0] + obs_dict['proprio'].shape[0],), dtype=np.float32)
