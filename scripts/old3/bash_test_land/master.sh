function runPrintStuffTimeLimit {
  chmod +x $1
  echo "running $1 for $2 seconds, and then killing"
  (./$1) & pid=$!
  echo "pid of $1 is $pid"
  (sleep $2 && kill -9 $pid)
  echo "Done."
}


runPrintStuffTimeLimit print_stuff0.sh 3

for seed in {1..4}
do
  # echo "copying "$seed
  cp -v print_stuff0.sh print_stuff$seed.sh
  print_stuff$seed.sh

  echo "copied print_stuff0.sh to print_stuff$seed.sh"
  echo "Number of instances of SEED=0 in print_stuff$seed.sh: $(grep "SEED=0" print_stuff$seed.sh | wc -l)"
  echo "Number of instances of SEED=$seed in print_stuff$seed.sh: $(grep "SEED=$seed" print_stuff$seed.sh | wc -l)"
  sed -i "s/SEED=0/SEED=$seed/" print_stuff$seed.sh
  echo "replaced SEED=0 with SEED=$seed"
  echo "Number of instances of SEED=0 in print_stuff$seed.sh: $(grep "SEED=0" print_stuff$seed.sh | wc -l)"
  echo "Number of instances of SEED=$seed in print_stuff$seed.sh: $(grep "SEED=$seed" print_stuff$seed.sh | wc -l)"

  runPrintStuffTimeLimit print_stuff$seed.sh 3
done
