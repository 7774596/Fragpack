#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Run one arm (arms.py) with its monitors, from the repository root.
#
# Waits until the GPUs have no compute processes, then runs the arm with the
# watchdog, the vLLM metrics sampler and an nvidia-smi log alongside. Aborts
# the arm if another user's process appears on its GPUs. Appends one status
# line per run to <out>/status and touches <out>/<arm>.done on success
# (a rerun skips arms that are done).
#
# Usage:
#   GPUS=0,1 TP=2 benchmarks/lossless_kv/run_arm.sh ARM OUT SCRATCH \
#       --model MODEL [arms.py options ...]
# Env: GPUS (PCI order, default 0), TP (default 1), VLLM_PORT (default 8047),
#      DATA_PATH (store filesystem for the watchdog's disk check; default
#      SCRATCH's parent).
set -u
[ $# -ge 3 ] || { sed -n '3,17p' "$0"; exit 2; }
ARM=$1; OUT=$(realpath -m "$2"); SCRATCH=$(realpath -m "$3"); shift 3
GPUS=${GPUS:-0}; TP=${TP:-1}; VLLM_PORT=${VLLM_PORT:-8047}
DATA_PATH=${DATA_PATH:-$(dirname "$SCRATCH")}
cd "$(dirname "$0")/../.." || exit 2
export CUDA_DEVICE_ORDER=PCI_BUS_ID VLLM_WORKER_MULTIPROC_METHOD=spawn
STATUS=$OUT/status
mkdir -p "$OUT" "$SCRATCH"
[ -e "$OUT/$ARM.done" ] && { echo "$ARM already done"; exit 0; }
# On btrfs, L2 files inherit NOCOW from the scratch directory.
chattr +C "$SCRATCH" 2>/dev/null

ME=$(id -un)
BUS=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader -i "$GPUS" \
  | cut -d: -f2- | paste -sd"|")
apps() {
  nvidia-smi --query-compute-apps=gpu_bus_id,pid --format=csv,noheader \
    | grep -iE "$BUS" | awk -F', ' '{print $2}'
}
foreign() {
  local p u
  for p in $(apps); do
    u=$(ps -o user= -p "$p" 2>/dev/null | tr -d ' ')
    [ -n "$u" ] && [ "$u" != "$ME" ] && echo "$p"
  done
}
idle() {
  local quiet=0
  while [ $quiet -lt 3 ]; do
    if [ -z "$(apps)" ]; then quiet=$((quiet + 1)); else quiet=0; fi
    sleep 30
  done
}

idle
echo "start $ARM $(git rev-parse --short HEAD)$(git diff --quiet || echo +dirty)" \
  "GPUS $GPUS TP $TP $* $(date '+%m-%d %T')" >> "$STATUS"
python -m benchmarks.lossless_kv.watchdog --output-dir "$OUT/$ARM" \
  --service-dir "$SCRATCH/$ARM" --vllm-url "http://127.0.0.1:$VLLM_PORT" \
  --data-path "$DATA_PATH" > "$OUT/$ARM.watchdog.log" 2>&1 &
WD=$!
python -m benchmarks.lossless_kv.metrics_sampler "$OUT/$ARM.metrics.tsv" \
  "http://127.0.0.1:$VLLM_PORT" 5 &
MS=$!
nvidia-smi --query-gpu=pci.bus_id,clocks.sm,utilization.gpu,memory.used,power.draw,temperature.gpu \
  --format=csv,noheader -i "$GPUS" -l 10 > "$OUT/$ARM.smi.csv" 2>&1 &
SMI=$!
python -m benchmarks.lossless_kv.arms --arms "$ARM" --artifact "$OUT" \
  --scratch "$SCRATCH" --gpus "$GPUS" --tensor-parallel-size "$TP" \
  --vllm-port "$VLLM_PORT" "$@" > "$OUT/$ARM.log" 2>&1 &
RUN=$!
RC=
while kill -0 $RUN 2>/dev/null; do
  f=$(foreign)
  if [ -n "$f" ]; then
    # arms.py stops its services and deletes the scratch store on SIGTERM.
    kill $RUN; RC=3
    echo "$ARM abort: foreign pid $f $(date +%T)" >> "$STATUS"
    break
  fi
  sleep 5
done
wait $RUN; rc=$?; [ -z "$RC" ] && RC=$rc
kill $WD $MS $SMI 2>/dev/null; wait $WD $MS $SMI 2>/dev/null
echo "$ARM rc=$RC $(tail -1 "$OUT/$ARM.log" | cut -c1-200) $(date '+%m-%d %T')" >> "$STATUS"
[ "$RC" = 0 ] && touch "$OUT/$ARM.done"
exit "$RC"
