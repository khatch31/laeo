import logging
from typing import Any, Callable, Mapping, Optional

from acme.utils.loggers import aggregators
from acme.utils.loggers import asynchronous as async_logger
from acme.utils.loggers import base
from acme.utils.loggers import csv
from acme.utils.loggers import filters
from acme.utils.loggers import terminal
from acme.utils.loggers import tf_summary

import os

from contrastive.wandb_logger import WANDBLoggerLabelWrapper, WANDBLogger

from copy import deepcopy


def make_default_logger(
    logdir: str,
    label: str,
    save_data: bool = True,
    time_delta: float = 1.0,
    asynchronous: bool = False,
    print_fn: Optional[Callable[[str], None]] = None,
    serialize_fn: Optional[Callable[[Mapping[str, Any]], str]] = base.to_numpy,
    steps_key: str = 'steps',
    wandblogger=None,
) -> base.Logger:
  """Makes a default Acme logger.

  Args:
    label: Name to give to the logger.
    save_data: Whether to persist data.
    time_delta: Time (in seconds) between logging events.
    asynchronous: Whether the write function should block or not.
    print_fn: How to print to terminal (defaults to print).
    serialize_fn: An optional function to apply to the write inputs before
      passing them to the various loggers.
    steps_key: Ignored.

  Returns:
    A logger object that responds to logger.write(some_dict).
  """
  del steps_key
  if not print_fn:
    print_fn = logging.info
  terminal_logger = terminal.TerminalLogger(label=label, print_fn=print_fn)

  loggers = [terminal_logger]

  if save_data:
    csv_logger = csv.CSVLogger(directory_or_file=logdir, label=label, add_uid=False)
    loggers.append(csv_logger)
    loggers.append(tf_summary.TFSummaryLogger(os.path.join(logdir, "tf_logs"), label=label))

    # new_wandblogger = WANDBLogger(os.path.join(deepcopy(wandblogger["logdir"]), label),
    #                                deepcopy(wandblogger["params"]),
    #                                deepcopy(wandblogger["group_name"]) + f"_{label}",
    #                                deepcopy(wandblogger["name"]),
    #                                deepcopy(wandblogger["project"]),
    #                                label=label)
    # loggers.append(new_wandblogger)

    if wandblogger is not None:
    #     print(f"\n\n\n\nwandblogger: {label}\n\n\n")
    #
    #     # logdir = "/iris/u/khatch/contrastive_rl/results/trash_results/offline_fetch_push-goals-no-noise/learner_goals/nonoise_collect_entropy-bc0.5_b1024/seed_0"
    #     # variant = {}
    #     # group_name = "nonoise_collect_entropy-bc0.5_b1024"
    #     # name = "seed_0"
    #     # project = "trash_results"
    #
    #     # new_wandblogger = WANDBLogger(logdir,
    #     #                           variant,
    #     #                           group_name,
    #     #                           name,
    #     #                           project)
    #
    #     # new_wandblogger = WANDBLogger(wandblogger["logdir"],
    #     #                               wandblogger["params"],
    #     #                               wandblogger["group_name"],
    #     #                               wandblogger["name"],
    #     #                               wandblogger["project"])
        # new_wandblogger = WANDBLogger(deepcopy(wandblogger["logdir"]),
        #                               deepcopy(wandblogger["params"]),
        #                               deepcopy(wandblogger["group_name"]),
        #                               deepcopy(wandblogger["name"]),
        #                               deepcopy(wandblogger["project"]))
    #     # loggers.append(WANDBLoggerLabelWrapper(new_wandblogger, label))
    #
        if label == "evaluator":
            wandbwrapper = WANDBLoggerLabelWrapper(wandblogger, label)
            loggers.append(wandbwrapper)

  # Dispatch to all writers and filter Nones and by time.
  logger = aggregators.Dispatcher(loggers, serialize_fn)
  logger = filters.NoneFilter(logger)
  if asynchronous:
    logger = async_logger.AsyncLogger(logger)
  logger = filters.TimeFilter(logger, time_delta)

  return logger
