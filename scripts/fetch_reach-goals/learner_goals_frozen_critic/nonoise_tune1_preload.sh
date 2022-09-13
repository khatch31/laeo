#!/bin/bash
#SBATCH --partition=iris-hi
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --job-name="crlgfcfetchreachgnonoise"
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
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco200/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin

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


python3 -u lp_contrastive_goals_frozen_critic.py \
--lp_launch_type=local_mt \
--project=contrastive_rl_goals \
--env_name=fetch_reach-goals-no-noise \
--entropy_coefficient=0 \
--description=nonoise_tune0.5_preload \
--max_number_of_steps=3000000 \
--batch_size=512 \
--actor_learning_rate=1e-4 \
--learning_rate=1e-4 \
--num_sgd_steps_per_step=32 \
--max_replay_size=10000000 \
--actor_min_std=0.1 \
--logdir=/iris/u/khatch/contrastive_rl/results \
--replay_buffer_load_dir=/iris/u/khatch/contrastive_rl/results/contrastive_rl_goals/fetch_reach-goals-no-noise/learner/nonoise_2/seed_0/checkpoints/replay_buffer \
--critic_checkpoint_path=/iris/u/khatch/contrastive_rl/results/contrastive_rl_goals/fetch_reach/learner/default/seed_0/checkpoints/learner

# --project=contrastive_rl_goals \
