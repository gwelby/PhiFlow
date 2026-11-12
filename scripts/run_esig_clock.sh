#!/bin/bash
# run_esig_clock.sh — one run of the clock-paced ENGINEERED-SIGNAL probe.
# prereg: QSOP/EVIDENCE/esig_clock_paced_prereg_2026-11-11.md
# Usage (on the surface, from a run dir holding phic, the probe, the producer
# and the scorer):   bash run_esig_clock.sh <run_dir> <ctl|wave>
# Exports sit on their own lines (the R5 export scar). One job at a time.
set -u
D="$1"
M="$2"
mkdir -p "$D/$M"
export SOMA_STATE_PATH="$D/$M/state.json"
export CUDA_VISIBLE_DEVICES=""
if [ "$M" = "ctl" ]; then
    PA="--constant 0.5"
elif [ "$M" = "wave" ]; then
    PA="--wave"
else
    echo "mode must be ctl or wave" >&2
    exit 2
fi
nice -n 19 python3 "$D/soma_signal_producer.py" $PA \
    --state-path "$SOMA_STATE_PATH" --interval 0.25 --duration 120 \
    > "$D/$M/prod.log" 2>&1 &
PP=$!
sleep 2
A=$(date +%s.%N)
nice -n 19 "$D/phic" "$D/type4_esig_clock_trace.phi" > "$D/$M/trace.txt" 2>&1
RC=$?
B=$(date +%s.%N)
kill "$PP" 2>/dev/null
wait "$PP" 2>/dev/null
python3 -c "import sys; a,b=float(sys.argv[1]),float(sys.argv[2]); print(f'rc=$RC wall_s={b-a:.3f} host=' + sys.argv[3])" \
    "$A" "$B" "$(hostname)" > "$D/$M/timing.txt"
python3 "$D/score_type4_trace.py" "$D/$M/trace.txt" --json > "$D/$M/score.json" 2>&1
cat "$D/$M/timing.txt"
cat "$D/$M/score.json"
