cd /iris/u/khatch/contrastive_rl/scripts/
echo "current directory"
pwd

maxSeconds=1500
# MAXSEED=5

function doGrepTests {
  echo "grep-ing in:"
  pwd

  echo 'grep -ri "batch_size=256" | wc -l'
  grep -ri "batch_size=256" | wc -l

  echo 'grep -ri "repr_dim=64" | wc -l'
  grep -ri "repr_dim=64" | wc -l

  echo 'grep -ri "hidden_layer_sizes=1024" | wc -l'
  grep -ri "hidden_layer_sizes=1024" | wc -l

  echo 'grep -ri "trash_results" | wc -l'
  grep -ri "trash_results" | wc -l

  echo 'grep -ri "contrastive_rl_goals10" | wc -l'
  grep -ri "contrastive_rl_goals10" | wc -l

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
}

cd offline_fetch_reach-goals-no-noise
doGrepTests
cd /iris/u/khatch/contrastive_rl/scripts/

cd offline_fetch_push-goals-no-noise
doGrepTests
cd /iris/u/khatch/contrastive_rl/scripts/


# function runWithTimeLimit {
#   chmod +x $1
#   echo "running $1 for $maxSeconds seconds, and then killing"
#   (./$1) & pid=$!
#   echo "pid of $1 is $pid"
#   sleep 120
#   pidscript=$(pgrep -f "python3 -u lp_contrastive_goals")
#   echo "pid of lp_contrastive_goals is $pidscript"
#
#   (sleep $maxSeconds && (kill -9 $pid && kill -9 $pidscript))
#   echo "Waiting 30 seconds..."
#   sleep 30
#   echo "Done."
# }
function runWithTimeLimit {
  chmod +x $1
  echo "running $1 for $maxSeconds seconds, and then killing"
  (./$1) & pid=$!
  echo "pid of $1 is $pid"
  (sleep $maxSeconds && kill -9 $pid)
  echo "Waiting 30 seconds..."
  sleep 30
  echo "Done."
}


function runMultipleSeeds {
  seed0File=$1

  echo "seed=0"
  runWithTimeLimit $seed0File

  for seed in {1..4}
  do
    yes '' | sed 3q
    echo "seed=$seed"
    newSeedFile=${seed0File:0:-8}"seed$seed.sh"
    echo "copying$seed0File to $newSeedFile"
    cp $seed0File $newSeedFile
    echo "Number of instances of seed=0 in $newSeedFile: $(grep "seed=0" $newSeedFile | wc -l)"
    sed -i "s/seed=0/seed=$seed/" $newSeedFile
    echo "replaced seed=0 with seed=$seed"
    echo "Number of instances of seed=0 in $newSeedFile: $(grep "seed=0" $newSeedFile | wc -l)"
    echo "Number of instances of seed=$seed in $newSeedFile: $(grep "seed=$seed" $newSeedFile | wc -l)"

    runWithTimeLimit $newSeedFile
  done
}

# TD3 PU
# runMultipleSeeds offline_fetch_reach-goals-no-noise/td3/nonoise_collect_alr=1e-5,clr=1e-5_minstd0.1_entropy/tune_ant_bc0.5_pu/seed0.sh
runMultipleSeeds offline_fetch_reach-goals-no-noise/td3/nonoise_collect_entropy/tune_ant_bc0.5_pu/seed0.sh

# TD3 SARSA BCE
runMultipleSeeds offline_fetch_reach-goals-no-noise/td3_sarsa/nonoise_collect_alr=1e-5,clr=1e-5_minstd0.1_entropy/tune_ant_bc0.5_bce/seed0.sh
runMultipleSeeds offline_fetch_reach-goals-no-noise/td3_sarsa/nonoise_collect_entropy/tune_ant_bc0.5_bce/seed0.sh

# TD3 SARSA PU
runMultipleSeeds offline_fetch_reach-goals-no-noise/td3_sarsa/nonoise_collect_alr=1e-5,clr=1e-5_minstd0.1_entropy/tune_ant_bc0.5_pu/seed0.sh
runMultipleSeeds offline_fetch_reach-goals-no-noise/td3_sarsa/nonoise_collect_entropy/tune_ant_bc0.5_pu/seed0.sh
