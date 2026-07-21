#!/usr/bin/env bash
set -euo pipefail

run_batch() {
  local config="$1"
  local result="$2"
  PYTHONPATH=src python scripts/run_stage4b_controls.py --config "$config" --overwrite
  PYTHONPATH=src python scripts/summarize_stage4b.py --results "$result"
}

run_batch configs/netket_stage4b_n8_component_replay.yaml results_stage4b/n8_component_replay
run_batch configs/netket_stage4b_n8_controls_core.yaml results_stage4b/n8_controls_core
run_batch configs/netket_stage4b_n8_capacity_compute.yaml results_stage4b/n8_capacity_compute
PYTHONPATH=src python scripts/summarize_stage4b_all.py --root results_stage4b
