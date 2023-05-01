import time
from typing import Optional

from absl import logging
from acme.utils.loggers import base

import wandb
import os

def _format_key(key: str) -> str:
  """Internal function for formatting keys in Tensorboard format."""
  return key.title().replace('_', '')


class WANDBLogger(base.Logger):
  """Logs to a tf.summary created in a given logdir.

  If multiple TFSummaryLogger are created with the same logdir, results will be
  categorized by labels.
  """

  def __init__(
      self,
      logdir,
      variant,
      group_name,
      name,
      project,
      # label,
      steps_key: Optional[str] = None
  ):
    """Initializes the logger.

    Args:
      logdir: directory to which we should log files.
      label: label string to use when logging. Default to 'Logs'.
      steps_key: key to use for steps. Must be in the values passed to write.
    """

    self._time = time.time()
    # self.label = label
    self._iter = 0
    # self.summary = tf.summary.create_file_writer(logdir)
    self._steps_key = steps_key

    dir = os.path.abspath(logdir)
    os.makedirs(dir, exist_ok=True)

    wandb.init(
        config=variant,
        project=project,
        dir=dir,
        id=group_name + "-" + name,
        settings=wandb.Settings(start_method="thread"),
        # settings=wandb.Settings(start_method="fork"),
        group=group_name,
        save_code=True,
        name=name,
        resume=None, # "allow",
        entity="laeo"
    )

  def write(self, values: base.LoggingData):
      raise NotImplementedError
  #   if self._steps_key is not None and self._steps_key not in values:
  #     logging.warn('steps key %s not found. Skip logging.', self._steps_key)
  #     return
  #
  #   step = values[self._steps_key] if self._steps_key is not None else self._iter
  #
  #   for key in values.keys() - [self._steps_key]:
  #       wandb.log({f'{self.label}/{_format_key(key)}': values[key]}, step=step)
  #       # print(f"{self.label}/{_format_key(key)}, step: {step}")
  #
  #   self._iter += 1

  def log(self, write_dict, step):
      wandb.log(write_dict, step=step)


      # if "learner/" in list(write_dict.keys())[0]:
      #     print(f"write_dict: {write_dict}, step: {step}")

  def close(self):
    pass

class WANDBLoggerLabelWrapper:
    def __init__(self, wandblogger, label, steps_key: Optional[str] = None):
        self.wandblogger = wandblogger
        self.label = label
        self._iter = 0
        self._steps_key = steps_key

    def write(self, values: base.LoggingData):
      if self._steps_key is not None and self._steps_key not in values:
        logging.warn('steps key %s not found. Skip logging.', self._steps_key)
        return

      step = values[self._steps_key] if self._steps_key is not None else self._iter

      # print(f"\n\n\n\nWrapper: {self.label}\n\n\n")

      for key in values.keys() - [self._steps_key]:
          self.wandblogger.log({f'{self.label}/{_format_key(key)}': values[key]}, step=step)

      self._iter += 1
