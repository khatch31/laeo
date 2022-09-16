#!/bin/bash
#SBATCH --partition=iris
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --job-name="nonoise_collect_alr=1e-5,clr=1e-5_minstd0.1"
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

python3 -u lp_contrastive.py \
--lp_launch_type=local_mt \
--project=contrastive_rl_goals3 \
--entropy_coefficient=0 \
--env_name=fetch_reach-goals-no-noise \
--description=nonoise_collect_alr=1e-5,clr=1e-5_minstd0.1 \
--save_data=true \
--actor_learning_rate=1e-5 \
--learning_rate=1e-5 \
--actor_min_std=0.1 \
--save_sim_state=true \
--num_actors=4 \
--max_checkpoints_to_keep=1000 \
--logdir=/iris/u/khatch/contrastive_rl/results






# --project=contrastive_rl_goals \
