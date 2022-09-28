cd /iris/u/khatch/contrastive_rl/scripts/
echo "current directory"
pwd

cd offline_fetch_reach-goals-no-noise
echo 'grep -ri "batch_size=256" | wc -l'
grep -ri "batch_size=256" | wc -l

echo 'grep -ri "repr_dim=64" | wc -l'
grep -ri "repr_dim=64" | wc -l

echo 'grep -ri "hidden_layer_sizes=1024" | wc -l'
grep -ri "hidden_layer_sizes=1024" | wc -l

echo 'grep -ri "trash_results" | wc -l'
grep -ri "trash_results" | wc -l

echo 'grep -ri "contrastive_rl_goals9" | wc -l'
grep -ri "contrastive_rl_goals9" | wc -l

echo 'grep -ri "lp_contrastive_goals.py" | wc -l'
grep -ri "lp_contrastive_goals.py" | wc -l

echo 'grep -ri "lp_contrastive_goals_td3.py" | wc -l'
grep -ri "lp_contrastive_goals_td3.py" | wc -l

echo 'grep -ri "offline_fetch_push-goals-no-noise" | wc -l'
grep -ri "offline_fetch_push-goals-no-noise" | wc -l

echo 'grep -ri "offline_fetch_reach-goals-no-noise" | wc -l'
grep -ri "offline_fetch_reach-goals-no-noise" | wc -l

echo 'grep -ri "recorded_data" | wc -l'
grep -ri "recorded_data" | wc -l

echo "grep -riF 'recorded_data \' | wc -l"
grep -riF 'recorded_data \' | wc -l
cd /iris/u/khatch/contrastive_rl/scripts/

function runSBATCHScriptOnWS {
  yes '' | sed 5q
  echo "========================================================"
  echo "running new SBATCH script on WS:"
  echo $1
  echo "========================================================"
  yes '' | sed 5q
  ./$1
  # ls $1
}

# BC
runSBATCHScriptOnWS offline_fetch_reach-goals-no-noise/bc/nonoise_collect_alr=1e-5,clr=1e-5_minstd0.1_entropy/tune_ant/seed2.sh
runSBATCHScriptOnWS offline_fetch_reach-goals-no-noise/bc/nonoise_collect_entropy/tune_ant/seed2.sh

# Learner Goals
runSBATCHScriptOnWS offline_fetch_reach-goals-no-noise/learner_goals/nonoise_collect_alr=1e-5,clr=1e-5_minstd0.1_entropy/tune_ant_bc0.5/seed2.sh
runSBATCHScriptOnWS offline_fetch_reach-goals-no-noise/learner_goals/nonoise_collect_entropy/tune_ant_bc0.5/seed2.sh

# TD3 BCE
runSBATCHScriptOnWS offline_fetch_reach-goals-no-noise/td3/nonoise_collect_alr=1e-5,clr=1e-5_minstd0.1_entropy/tune_ant_bc0.5_bce/seed2.sh
runSBATCHScriptOnWS offline_fetch_reach-goals-no-noise/td3/nonoise_collect_entropy/tune_ant_bc0.5_bce/seed2.sh

# TD3 PU
runSBATCHScriptOnWS offline_fetch_reach-goals-no-noise/td3/nonoise_collect_alr=1e-5,clr=1e-5_minstd0.1_entropy/tune_ant_bc0.5_pu/seed2.sh
runSBATCHScriptOnWS offline_fetch_reach-goals-no-noise/td3/nonoise_collect_entropy/tune_ant_bc0.5_pu/seed2.sh

# TD3 SARSA BCE
runSBATCHScriptOnWS offline_fetch_reach-goals-no-noise/td3_sarsa/nonoise_collect_alr=1e-5,clr=1e-5_minstd0.1_entropy/tune_ant_bc0.5_bce/seed2.sh
runSBATCHScriptOnWS offline_fetch_reach-goals-no-noise/td3_sarsa/nonoise_collect_entropy/tune_ant_bc0.5_bce/seed2.sh

# TD3 SARSA PU
runSBATCHScriptOnWS offline_fetch_reach-goals-no-noise/td3_sarsa/nonoise_collect_alr=1e-5,clr=1e-5_minstd0.1_entropy/tune_ant_bc0.5_pu/seed2.sh
runSBATCHScriptOnWS offline_fetch_reach-goals-no-noise/td3_sarsa/nonoise_collect_entropy/tune_ant_bc0.5_pu/seed2.sh
