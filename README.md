# Contrastive Example-Based Control

<p align="center"> Kyle Hatch, &nbsp; Benjamin Eysenbach, &nbsp; Rafael Rafailov &nbsp; Tianhe Yu &nbsp; <br> Ruslan Salakhutdinov &nbsp; Sergey Levine &nbsp; Chelsea Finn </p>


<p align="center">
   <a href="https://sites.google.com/view/laeo-rl">website &nbsp </a>
   <a href="https://arxiv.org/abs/2307.13101">paper</a>
</p>
<!-- ![diagram of contrastive RL](contrastive_rl.png) -->

*Abstract*: While many real-world problems that might benefit from reinforcement learning, these problems rarely fit into the MDP mold: interacting with the environment is often expensive and specifying reward functions is challenging. Motivated by these challenges, prior work has developed data-driven approaches that learn entirely from samples from the transition dynamics and examples of high-return states. These methods typically learn a reward function from high-return states, use that reward function to label the transitions, and then apply an offline RL algorithm to these transitions. While these methods can achieve good results on many tasks, they can be complex, often requiring regularization and temporal difference updates. In this paper, we propose a method for offline, example-based control that learns an implicit model of multi-step transitions, rather than a reward function. We show that this implicit model can represent the Q-values for the example-based control problem. Across a range of state-based and image-based offline control tasks, our method outperforms baselines that use learned reward functions; additional experiments demonstrate improved robustness and scaling with dataset size.


![alt text](https://github.com/[username]/[reponame]/blob/[branch]/images/fetch-reach.gif?raw=true)



This repository contains the new algorithms, the baselines, and the associated environments used in this paper. If you use this repository, please consider adding the following citation:



```
@inproceedings{hatch2023contrastive,
  title={Contrastive Example-Based Control},
  author={Hatch, Kyle Beltran and Eysenbach, Benjamin and Rafailov, Rafael and Yu, Tianhe and Salakhutdinov, Ruslan and Levine, Sergey and Finn, Chelsea},
  booktitle={Learning for Dynamics and Control Conference},
  pages={155--169},
  year={2023},
  organization={PMLR}
}
```
### Installation

1. Create an Anaconda environment: `conda create -n laeo python=3.9
   -y`
2. Activate the environment: `conda activate laeo`
3. Install the dependencies: `pip install -r requirements.txt --no-deps`
<!-- 4. Check that the installation worked: `./run.sh` -->

<!-- ### Running the experiments

To check that the installation has completed, run `./run.sh` to perform training for just a handful of steps. To replicate the results from the paper, please run:
```python lp_contrastive.py```

Check out the `lp_contrastive.py` file for more information on how to select different algorithms and environments. For example, to try the offline RL experiments, set `env_name = 'offline_ant_umaze'`. One important note is that the image-based experiments should be run using multiprocessing, to avoid OpenGL context errors:
```python lp_contrastive.py --lp_launch_type=local_mp``` -->
### Running the experiments

To replicate the results from the paper, please run:
```python3 lp_contrastive_goals.py```

Check out the `lp_contrastive_goals.py` file for more information on how to select different algorithms and environments.
```
unset LD_LIBRARY_PATH
unset LD_PRELOAD
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco200/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-000
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export MUJOCO_GL="egl"
export XLA_PYTHON_CLIENT_MEM_FRACTION=.7

python3 -u lp_contrastive_goals.py \
--lp_launch_type=local_mt \
--project=my_description \
--env_name=offline_fetch_push-goals-no-noise \
--seed=0 \
--description=state_push \
--entropy_coefficient=0 \
--max_number_of_steps=500000 \
--repr_dim=256 \
--hidden_layer_sizes=1024 \
--hidden_layer_sizes=1024 \
--max_replay_size=10000000 \
--actor_min_std=0.1 \
--batch_size=1024 \
--num_sgd_steps_per_step=1 \
--num_actors=0 \
--twin_q=true \
--bc_coef=0.5 \
--logdir=/path/to/results/ \
--data_load_dir=/path/to/data/fetch/push/medium_replay_10_seeds
```

For image based experiments, run:

```
unset LD_LIBRARY_PATH
unset LD_PRELOAD
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco200/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-000
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export MUJOCO_GL="egl"
export XLA_PYTHON_CLIENT_MEM_FRACTION=.7

python3 -u lp_contrastive_goals.py \
--lp_launch_type=local_mt \
--project=my_wandb_project \
--env_name=offline_fetch_push_image_minimal-goals-no-noise \
--seed=0 \
--description=my_description \
--entropy_coefficient=0 \
--max_number_of_steps=500000 \
--repr_dim=256 \
--hidden_layer_sizes=1024 \
--hidden_layer_sizes=1024 \
--max_replay_size=10000000 \
--actor_min_std=0.1 \
--batch_size=1024 \
--num_sgd_steps_per_step=1 \
--prefetch_size=1 \
--num_parallel_calls=1 \
--num_actors=0 \
--twin_q=true \
--bc_coef=0.5 \
--logdir=/path/to/results/ \
--data_load_dir=/path/to/data/fetch/push/medium_replay_10_seeds
```

### Data
Datasets will be made available soon.

### Questions?
If you have any questions, comments, or suggestions, please reach out to Kyle Hatch (khatch@stanford.edu, khatch@cs.stanford.edu).
