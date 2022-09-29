#!/bin/bash
#SBATCH --partition=iris
#SBATCH --time=9:00:00
#SBATCH --nodes=1
#SBATCH --job-name="nonoise_collect_entropy--ta_bc0.5"
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --exclude=iris4,iris5,iris6


which python3
cd /iris/u/khatch/contrastive_rl
pwd
source ~/.bashrc
# conda init bash
source /iris/u/khatch/anaconda3/bin/activate
conda activate crl2

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

python3 -u lp_contrastive_goals.py \
--lp_launch_type=local_mt \
--project=contrastive_rl_goals11 \
--env_name=offline_fetch_reach_image-goals-no-noise \
--seed=1 \
--description=nonoise_collect_entropy--ta \
--use_gcbc=true \
--entropy_coefficient=0 \
--max_number_of_steps=7000 \
--repr_dim=256 \
--hidden_layer_sizes=1024 \
--hidden_layer_sizes=1024 \
--max_replay_size=10000000 \
--actor_min_std=0.1 \
--batch_size=256 \
--num_actors=0 \
--twin_q=true \
--bc_coef=0.5 \
--logdir=/iris/u/khatch/contrastive_rl/results \
--data_load_dir=/iris/u/khatch/contrastive_rl/results/contrastive_rl_goals3/fetch_reach-goals-no-noise/learner/nonoise_collect_entropy/seed_0/recorded_data
# --data_load_dir=datasets/reach_easy
