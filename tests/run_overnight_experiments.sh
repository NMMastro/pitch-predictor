#!/usr/bin/env bash

# Run a small overnight experiment grid for the pitch transformer.
#
# Each run trains for the requested number of epochs, appends its metrics to
# results/scoreboard.jsonl, and copies its best checkpoint into a unique output
# directory so a later run cannot overwrite it.

set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

python_bin="$project_dir/.venv/bin/python"
clean_data="$project_dir/data/clean/pitches.parquet"

if [[ ! -x "$python_bin" ]]; then
    echo "Missing project Python environment: $python_bin" >&2
    exit 1
fi

if [[ ! -f "$clean_data" ]]; then
    echo "Missing cleaned pitch data: $clean_data" >&2
    exit 1
fi

# Override these from the command line if you want a smaller or larger grid.
epochs="${EPOCHS:-20}"
read -r -a windows <<< "${WINDOWS:-64 128 256}"
read -r -a seeds <<< "${SEEDS:-1 2}"

run_id="$(date +%Y%m%d_%H%M%S)"
output_dir="$project_dir/results/overnight/$run_id"
log_dir="$output_dir/logs"
checkpoint_dir="$output_dir/checkpoints"

mkdir -p "$log_dir" "$checkpoint_dir"

echo "Overnight experiment: $run_id"
echo "Epochs: $epochs"
echo "Windows: ${windows[*]}"
echo "Seeds: ${seeds[*]}"
echo "Output: $output_dir"

for window in "${windows[@]}"; do
    for seed in "${seeds[@]}"; do
        name="window_${window}_seed_${seed}"
        log_file="$log_dir/$name.log"
        source_checkpoint="$project_dir/results/checkpoints/pitch_type_w${window}.pt"
        saved_checkpoint="$checkpoint_dir/pitch_type_w${window}_seed${seed}.pt"

        echo
        echo "Starting $name"

        "$python_bin" src/model/train.py \
            --epochs "$epochs" \
            --window "$window" \
            --seed "$seed" \
            2>&1 | tee "$log_file"

        cp "$source_checkpoint" "$saved_checkpoint"
        echo "Saved checkpoint: $saved_checkpoint"
    done
done

echo
echo "All experiments finished."
echo "Metrics: $project_dir/results/scoreboard.jsonl"
echo "Logs and checkpoints: $output_dir"
