#!/bin/bash
# Paper-1 experiments: does admissible (Euclidean) distance shaping HURT a kinodynamic
# robot, and does the harm scale with the constraint knobs?
#
# Usage:
#   bash experiments/run_theory.sh pilot       # 1 seed, short — de-risk the effect fast
#   bash experiments/run_theory.sh E1          # core: shaping sweep, seeds 0-2
#   bash experiments/run_theory.sh E2          # curvature (omega) sweep
#   bash experiments/run_theory.sh E3          # momentum (a_max) sweep
#   bash experiments/run_theory.sh E4          # open-world control
#
# Results appended to experiments/results.csv: tag,env,shaping,omega,a_max,seed,steps,det_sr,stoch_sr
set -u
cd /home/rr/kinodynamic_rl
source .venv/bin/activate 2>/dev/null
OUT=experiments/results.csv
[ -f "$OUT" ] || echo "tag,env,shaping,omega,a_max,seed,steps,det_sr,stoch_sr" > "$OUT"

train_eval () {
  TAG=$1; ENV=$2; SHAPE=$3; OMEGA=$4; AMAX=$5; SEED=$6; STEPS=$7
  EXTRA=""
  [ "$OMEGA" != "-" ] && EXTRA="$EXTRA env.omega_max_override=$OMEGA"
  [ "$AMAX"  != "-" ] && EXTRA="$EXTRA env.a_max_override=$AMAX"
  python train.py env=$ENV shaping=$SHAPE init=fixed network=mlp obs=full_state \
    train.timesteps=$STEPS train.num_envs=16 train.seed=$SEED train.max_lr=6e-5 \
    $EXTRA > /tmp/th_${TAG}_${SHAPE}_${SEED}.log 2>&1
  RUN=$(ls -dt runs/${SHAPE}_mlp_full_state/*/ | head -1)
  CK=$(ls -t $RUN/checkpoints/agent_*.pt | head -1)
  L=$(PYTHONPATH=. python scripts/fasteval.py eval.checkpoint=$CK env=$ENV shaping=$SHAPE \
      init=fixed network=mlp obs=full_state $EXTRA eval.episodes=30 2>/dev/null)
  DET=$(echo "$L"|awk '/deterministic/{for(i=1;i<=NF;i++)if($i~/success=/){gsub(/[^0-9.]/,"",$(i+1));print $(i+1)}}')
  STO=$(echo "$L"|awk '/stochastic/{for(i=1;i<=NF;i++)if($i~/success=/){gsub(/[^0-9.]/,"",$(i+1));print $(i+1)}}')
  echo "$TAG,$ENV,$SHAPE,$OMEGA,$AMAX,$SEED,$STEPS,$DET,$STO" | tee -a "$OUT"
}

case "${1:-pilot}" in
  pilot)
    for S in none euclidean dijkstra; do train_eval pilot theory_corridor $S - - 0 250000; done ;;
  E1)
    for SEED in 0 1 2; do for S in none euclidean dijkstra; do
      train_eval E1 theory_corridor $S - - $SEED 400000; done; done ;;
  E2)
    for OM in 3.14159 1.5708 1.0472 0.5236; do for S in none euclidean dijkstra; do
      train_eval E2 theory_corridor $S $OM - 0 400000; done; done ;;
  E3)
    for AM in 8 4 2 1; do for S in none euclidean dijkstra; do
      train_eval E3 theory_corridor $S - $AM 0 400000; done; done ;;
  E4)
    for SEED in 0 1 2; do for S in none euclidean dijkstra; do
      train_eval E4 theory_open $S - - $SEED 400000; done; done ;;
  *) echo "unknown: $1"; exit 1 ;;
esac
echo "DONE $1"
