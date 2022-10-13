import os
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm, trange
from collections import OrderedDict, defaultdict

# BASEDIR = ""

PROJECT_DIRS = [
    "/iris/u/khatch/contrastive_rl/results/contrastive_rl_goals11",
    "/iris/u/khatch/contrastive_rl/results/contrastive_rl_goals9",
    "/iris/u/khatch/contrastive_rl/results/contrastive_rl_goals5",
    "/iris/u/rafailov/ICLR2023/contrastive_rl_goals11",
]

# TASKS = ["offline_fetch_push-goals-no-noise", "offline_fetch_reach-goals-no-noise",
#         "offline_fetch_push_image-goals-no-noise", "offline_fetch_reach_image-goals-no-noise"]

TASKS_AND_DATASETS = [("offline_fetch_push-goals-no-noise", "nonoise_collect_entropy"),
          ("offline_fetch_push-goals-no-noise", "nonoise_collect_alr=1e-5,clr=1e-4_minstd0.1_entropy"),
          ("offline_fetch_reach-goals-no-noise", "nonoise_collect_entropy"),
          ("offline_fetch_reach-goals-no-noise", "nonoise_collect_alr=1e-5,clr=1e-5_minstd0.1_entropy"),

        ("offline_fetch_push_image-goals-no-noise", "nonoise_collect_entropy"),
        # ("offline_fetch_push_image-goals-no-noise", "nonoise_collect_alr=1e-5,clr=1e-4_minstd0.1_entropy"),
        ("offline_fetch_reach_image-goals-no-noise", "nonoise_collect_entropy"),
        ("offline_fetch_reach_image-goals-no-noise", "nonoise_collect_alr=1e-5,clr=1e-5_minstd0.1_entropy"),]

SKIPFILES = [
    "/iris/u/khatch/contrastive_rl/results/contrastive_rl_goals11/offline_fetch_push-goals-no-noise/learner_goals/nonoise_collect_entropy--tune_ant_bc0.5/seed_0/logs/evaluator/logs.csv",
    "/iris/u/khatch/contrastive_rl/results/contrastive_rl_goals11/offline_fetch_reach-goals-no-noise/td3/nonoise_collect_entropy--tune_ant_bc0.5_bce/seed_3/logs/evaluator/logs.csv",
    "/iris/u/khatch/contrastive_rl/results/contrastive_rl_goals11/offline_fetch_push_image-goals-no-noise/bc/nonoise_collect_entropy--ta/seed_2/logs/evaluator/logs.csv",
    "/iris/u/khatch/contrastive_rl/results/contrastive_rl_goals11/offline_fetch_push_image-goals-no-noise/bc/nonoise_collect_entropy--ta/seed_1/logs/evaluator/logs.csv",
    "/iris/u/khatch/contrastive_rl/results/contrastive_rl_goals11/offline_fetch_push_image-goals-no-noise/td3/nonoise_collect_entropy--ta_bc0.5_pu/seed_1/logs/evaluator/logs.csv",
    "/iris/u/khatch/contrastive_rl/results/contrastive_rl_goals11/offline_fetch_push_image-goals-no-noise/td3/nonoise_collect_entropy--ta_bc0.5_pu/seed_2/logs/evaluator/logs.csv",
    "/iris/u/khatch/contrastive_rl/results/contrastive_rl_goals5/offline_fetch_reach-goals-no-noise/learner_goals/nonoise_collect_alr=1e-5,clr=1e-5_minstd0.1_entropy--tune_ant_bc0.5/seed_0/logs/evaluator/logs.csv",
    '/iris/u/khatch/contrastive_rl/results/contrastive_rl_goals5/offline_fetch_reach-goals-no-noise/learner/nonoise_collect_alr=1e-5,clr=1e-5_minstd0.1_entropy--tune_ant_bc0.5/seed_0/logs/evaluator/logs.csv',
    # "/iris/u/khatch/contrastive_rl/results/contrastive_rl_goals5/offline_fetch_reach-goals-no-noise/learner/nonoise_collect_alr=1e-5,clr=1e-5_minstd0.1_entropy--tune_ant_bc0.5/seed_0/logs/evaluator/logs.csv",


    # Duplicate seeds?
    # '/iris/u/khatch/contrastive_rl/results/contrastive_rl_goals9/offline_fetch_push-goals-no-noise/td3/nonoise_collect_entropy--tune_ant_bc0.5_pu/seed_1/logs/evaluator/logs.csv',
    # '/iris/u/khatch/contrastive_rl/results/contrastive_rl_goals9/offline_fetch_push-goals-no-noise/bc/nonoise_collect_entropy--tune_ant/seed_1/logs/evaluator/logs.csv',
    # '/iris/u/khatch/contrastive_rl/results/contrastive_rl_goals9/offline_fetch_push-goals-no-noise/bc/nonoise_collect_entropy--tune_ant/seed_2/logs/evaluator/logs.csv',


    "/iris/u/khatch/contrastive_rl/results/contrastive_rl_goals9/offline_fetch_push-goals-no-noise/learner_goals/nonoise_collect_alr=1e-5,clr=1e-4_minstd0.1_entropy--tune_ant_bc0.5/seed_0/logs/evaluator/logs.csv",
    "/iris/u/khatch/contrastive_rl/results/contrastive_rl_goals5/offline_fetch_push-goals-no-noise/learner_goals/nonoise_collect_entropy--tune_ant_bc0.5/seed_0/logs/evaluator/logs.csv",
    "/iris/u/khatch/contrastive_rl/results/contrastive_rl_goals5/offline_fetch_push-goals-no-noise/learner_goals/nonoise_collect_alr=1e-5,clr=1e-4_minstd0.1_entropy--tune_ant_bc0.5/seed_0/logs/evaluator/logs.csv",
    "/iris/u/khatch/contrastive_rl/results/contrastive_rl_goals5/offline_fetch_reach-goals-no-noise/learner_goals/nonoise_collect_entropy--tune_ant_bc0.5/seed_0/logs/evaluator/logs.csv",
]


def map_dataset_names(name):
    if "collect_entropy" in name:
        return "easy"

    if "1e-5" in name:
        return "hard"

def task_algo_dataset_seed_from_logfile(dir, logfile):
    x = logfile[len(dir) + 1:-len("/logs/evaluator/logs.csv")]
    x = x.split("/")
    algo = x[0]
    dataset = map_dataset_names(x[1].split("--")[0])
    seed = int(x[2].split("_")[-1])

    fulltask = dir.split("/")[-1]
    task = get_task_from_fulltask(fulltask)

    if algo == "td3":
        if "bce" in logfile:
            algo += "_bce"
        elif "pu" in logfile:
            algo += "_pu"
        else:
            raise ValueError()


    return task, algo, dataset, seed

def get_task_from_fulltask(fulltask):
    if "push" in fulltask:
        task = "push"
    elif "reach" in fulltask:
        task = "reach"

    if "image" in fulltask:
        task += "_image"

    return task

def rec_dd():
    return defaultdict(rec_dd)

def skip_file(dir, logfile, task, algo, dataset, seed):
    skip = False
    if logfile in SKIPFILES:
        skip = True

    if algo != "bc" and "bc0.5" not in logfile:
        # print(f"Skipping because \"algo != \"bc\" and \"bc0.5\" not in logfile\": \"{logfile}\"")
        skip = True

    if "td" in algo and "td3" not in algo:
        # print(f"Skipping because \"\"td\" in algo and \"td3\" not in algo\" not in logfile\": \"{logfile}\"")
        skip = True

    if "sarsa" in logfile:
        skip = True

    if "contrastive_rl_goals11" not in logfile and "seed_0" not in logfile:
        skip = True

    # if skip:
    #     print("Skipped")

    return skip

def process():
    master_dict = rec_dd()
    for project_dir in PROJECT_DIRS:
        # for task in TASKS:
        for task_full, dataset_full in TASKS_AND_DATASETS:
            # dir = os.path.join(BASEDIR, project_dir, task_full)
            dir = os.path.join(project_dir, task_full)
            logfiles = glob(os.path.join(dir, "**", dataset_full + "*", "**", "logs", "evaluator", "logs.csv"), recursive=True)

            for logfile in logfiles:
                # if "rafailov" in logfile:
                #     print(logfile)
                task, algo, dataset, seed = task_algo_dataset_seed_from_logfile(dir, logfile)
                if skip_file(dir, logfile, task, algo, dataset, seed):
                    continue

                success_1000 = load_single_log(logfile)

                if seed in master_dict[task + "_" + dataset][algo]:
                    # print("\nSeed already here")
                    # print(f"master_dict[{task + '_' + dataset}][{algo}].keys():", master_dict[task + "_" + dataset][algo].keys())
                    # print(f"\"{logfile}\"" + ",")
                    print(f"\nTried to add seed {seed}, but master_dict[{task + '_' + dataset}][{algo}].keys() = {master_dict[task + '_' + dataset][algo].keys()}")
                    print(f"\"{logfile}\"" + ",")

                master_dict[task + "_" + dataset][algo][seed] = {"final_success_1000":success_1000[-1]}

    for task_dataset in master_dict.keys():
        print(task_dataset)
        for algo in master_dict[task_dataset].keys():
            final_success_1000_arr = np.array([master_dict[task_dataset][algo][key]["final_success_1000"] for key in master_dict[task_dataset][algo].keys()])
            print(f"\t{algo}: mean: {np.mean(final_success_1000_arr):.3f} |  std: {np.std(final_success_1000_arr):.3f} n_seeds: {len(master_dict[task_dataset][algo].keys())}")


def load_single_log(logfile):
    header = get_header(logfile)
    try:
        data = np.loadtxt(logfile, delimiter=",", skiprows=1)
    except:
        print(f"Issue with np.loadtxt on \"{logfile}\".")
        import pdb; pdb.set_trace()
    # learner_steps = data[:, header.index("learner_steps")]
    # success_1000 = np.nan_to_num(data[:, header.index("success_1000")])
    # return learner_steps, success_1000
    # learner_steps = data[:, header.index("learner_steps")]
    success_1000 = np.nan_to_num(data[:, header.index("success_1000")])
    return success_1000

def get_header(logfile):
    with open(logfile, "r") as f:
        # try:
        #     line = f.readline()
        # except:
        #     import pdb; pdb.set_trace()
        line = next(f)
        header = line.replace("\n", "").split(",")
        return header

if __name__ == "__main__":
    process()
