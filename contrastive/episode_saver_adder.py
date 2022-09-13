from typing import Callable, Optional, Iterable, Tuple

from acme import specs
from acme import types
from acme.adders.reverb import base
from acme.adders.reverb import utils

import dm_env
import numpy as np
import reverb
import tensorflow as tf
import tree

_PaddingFn = Callable[[Tuple[int, ...], np.dtype], np.ndarray]

from acme.adders import reverb as adders_reverb
import os
import io
from glob import glob

class EpisodeAdderSaver(adders_reverb.EpisodeAdder):
  def __init__(self, *args, save=False,  savedir=None, **kwargs):
      self._saved_ep_idx = 1
      if save:
          if os.path.isdir(savedir):
              episode_files = glob(os.path.join(savedir, "*.npz"))
              get_ep_no = lambda x:int(x.split("/")[-1].split("_")[0].split("-")[-1])
              episode_files = sorted(episode_files, key=get_ep_no)
              self._saved_ep_idx = get_ep_no(episode_files[-1]) + 1
          else:
              os.makedirs(savedir, exist_ok=True)

      self._save = save
      self._savedir = savedir
      super().__init__(*args, **kwargs)

  def _write_last(self):
    episode_steps = self._writer.episode_steps
    if self._padding_fn is not None and self._writer.episode_steps < self._max_sequence_length:
      history = self._writer.history
      padding_step = dict(
          observation=history['observation'],
          action=history['action'],
          reward=history['reward'],
          discount=history['discount'],
          extras=history.get('extras', ()))
      # Get shapes and dtypes from the last element.
      padding_step = tree.map_structure(
          lambda col: self._padding_fn(col[-1].shape, col[-1].dtype),
          padding_step)
      padding_step['start_of_episode'] = False

      while self._writer.episode_steps < self._max_sequence_length:
        self._writer.append(padding_step)

    trajectory = tree.map_structure(lambda x: x[:], self._writer.history)

    # Pack the history into a base.Step structure and get numpy converted
    # variant for priotiy computation.
    trajectory = base.Trajectory(**trajectory)

    if self._save:
        episode = {}
        episode["observation"] = trajectory.observation.numpy()
        episode["action"] = trajectory.action.numpy()
        episode["reward"] = trajectory.reward.numpy()
        episode["discount"] = trajectory.discount.numpy()
        episode["extras"] = trajectory.extras
        episode["start_of_episode"] = trajectory.start_of_episode.numpy()

        # Trim off padding steps
        if episode_steps < episode["observation"].shape[0]:
            episode = {key: val[:episode_steps] for key, val in episode.items()}

        length = len(episode['reward'])
        filename = os.path.join(self._savedir, f'ep-{self._saved_ep_idx}_len-{length}.npz')

        if os.path.exists(filename):
            raise ValueError(f"\"{filename}\" already exists.")

        with io.BytesIO() as f1:
            np.savez_compressed(f1, **episode)
            f1.seek(0)
            with open(filename, 'wb') as f2:
                f2.write(f1.read())

        self._saved_ep_idx += 1



    # Calculate the priority for this episode.
    table_priorities = utils.calculate_priorities(self._priority_fns,
                                                  trajectory)

    # Create a prioritized item for each table.
    for table_name, priority in table_priorities.items():
      self._writer.create_item(table_name, priority, trajectory)
      self._writer.flush(self._max_in_flight_items)
