#!/usr/bin/env bash
# Runs the remaining experiments in dependency order after E5; E3 last (wall-clock timings, alone).
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
while pgrep -f "^.venv/bin/python -u -m experiments.e05" >/dev/null; do sleep 30; done
for e in e07_online e04_saa e09_sensitivity e06_pareto e08_network_design; do
  echo "=== $e start $(date)"; $PY -u -m experiments.$e > results/$e.log 2>&1; echo "=== $e exit $? $(date)"
done
echo "=== e03 start $(date)"; $PY -u -m experiments.e03_scaling a b c d > results/e03.log 2>&1; echo "=== e03 exit $? $(date)"
