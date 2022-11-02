#!/bin/bash
#SBATCH --partition=iris
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --job-name="nonoise_collect_entropy"
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --account=iris

which python3
cd /iris/u/khatch/contrastive_rl
pwd
source ~/.bashrc
# conda init bash
source /iris/u/khatch/anaconda3/bin/activate
conda activate contrastive_rl

unset LD_LIBRARY_PATH
unset LD_PRELOAD
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco200/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-000
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

echo $SLURM_JOB_GPUS
export GPUS=$SLURM_JOB_GPUS
export MUJOCO_GL="egl"
export XLA_PYTHON_CLIENT_MEM_FRACTION=.7

which python3
nvidia-smi
pwd
ls -l /usr/local

# export CUDA_VISIBLE_DEVICES=""
python3 -u gpu_test.py

which python3
nvidia-smi
pwd
ls -l /usr/local

python3 -u lp_contrastive.py \
--lp_launch_type=local_mt \
--project=contrastive_rl_goals15 \
--env_name=fetch_push-goals-no-noise \
--description=nonoise_collect_entropy \
--seed=10 \
--save_data=true \
--save_sim_state=true \
--num_actors=4 \
--max_checkpoints_to_keep=1000 \
--logdir=/iris/u/khatch/contrastive_rl/results
