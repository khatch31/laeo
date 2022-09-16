#!/bin/bash
#SBATCH --partition=iris-hi
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --job-name="nonoise_collect--tune_ant_bc0.5"
#SBATCH --gres=gpu:1
#SBATCH --mem=32G


which python3
cd /iris/u/khatch/contrastive_rl
pwd
source /sailhome/khatch/.bashrc
conda init bash
source /iris/u/khatch/anaconda3/bin/activate
conda activate contrastive_rl

unset LD_LIBRARY_PATH
unset LD_PRELOAD
# export LD_PRELOAD=$LD_PRELOAD:/usr/lib/x86_64-linux-gnu/libGLEW.so.1.13.0
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-000
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/sailhome/khatch/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/sailhome/khatch/.mujoco/mujoco200/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/sailhome/khatch/.mujoco/mujoco210/bin

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:=/iris/u/khatch/anaconda3/envs/contrastive_rl/lib/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-000

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

echo $SLURM_JOB_GPUS
export GPUS=$SLURM_JOB_GPUS

export MUJOCO_GL="egl"
# unset LD_PRELOAD

which python3
nvidia-smi
pwd
ls -l /usr/local

# export XLA_PYTHON_CLIENT_PREALLOCATE="false"
# export XLA_PYTHON_CLIENT_MEM_FRACTION=".1"

python3 -u lp_contrastive_goals.py \
--lp_launch_type=local_mt \
--project=contrastive_rl_goals5 \
--env_name=offline_fetch_reach-goals-no-noise \
--description=nonoise_collect--tune_ant_bc0.5 \
--entropy_coefficient=0 \
--max_number_of_steps=1000000 \
--actor_learning_rate=1e-4 \
--learning_rate=1e-4 \
--repr_dim=256 \
--hidden_layer_sizes=1024 \
--hidden_layer_sizes=1024 \
--max_replay_size=10000000 \
--actor_min_std=0.1 \
--batch_size=1024 \
--num_actors=0 \
--twin_q=true \
--bc_coef=0.5 \
--logdir=/iris/u/khatch/contrastive_rl/results \
--data_load_dir=/iris/u/khatch/contrastive_rl/results/contrastive_rl_goals3/fetch_reach-goals-no-noise/learner/nonoise_collect/seed_0/recorded_data
