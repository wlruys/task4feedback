#!/usr/bin/env bash
# run_parallel_launcher.sh
# -----------------------------------------------------------------------------
# Launch commands from a file using GNU parallel.
# Optionally pins each concurrent job to a *disjoint group* of K CPU cores.
#
# Usage:
#   ./run_parallel_launcher.sh <commands.txt> <K> [prefix] [out_dir]
#
# Notes:
# - Requires: GNU parallel (https://www.gnu.org/software/parallel/)
# - CPU pinning uses 'taskset' (Linux). On macOS, pinning is skipped automatically.
# - Logs:
#     * Structured per-job stdout/stderr under:  [out_dir]/results/...
#     * A TSV job log at:                        [out_dir]/manifest.tsv
# - This script is single-node by design. For multi-node, run once per node and
#   merge the per-node manifest TSVs afterward.
#
# Environment overrides:
#   CORES="0-15,32-47"          # restrict the machine's core pool (Linux/macOS)
#   PREAMBLE=("cmd1" "cmd2")    # bash snippets run before each command (array)
#   PARALLEL_OPTS="--bar"       # extra flags passed to 'parallel' (e.g., --bar)
# -----------------------------------------------------------------------------

set -euo pipefail

commands_file="${1:-commands.txt}"
k_per_job="${2:-1}"
prefix="${3:-job}"
out_dir="${4:-logs/parallel}"

[[ -f "$commands_file" ]] || { echo "Missing commands file: $commands_file" >&2; exit 1; }
[[ "$k_per_job" =~ ^[0-9]+$ && "$k_per_job" -ge 1 ]] || { echo "K must be a positive integer" >&2; exit 1; }
command -v parallel >/dev/null 2>&1 || { echo "GNU parallel not found" >&2; exit 1; }

mkdir -p "$out_dir"
results_dir="$out_dir/results"
groups_file="$out_dir/core_groups.txt"
manifest_tsv="$out_dir/manifest.tsv"

OS=$(uname -s || echo "Unknown")
NODE=$(hostname -s 2>/dev/null || hostname || echo "unknown-node")

# --- Helpers ------------------------------------------------------------------

expand_cpulist() {
  # Expand "0-3,5,7-9" -> one number per line.
  local list="${1:-}" IFS=',' part
  for part in $list; do
    if [[ "$part" =~ ^([0-9]+)-([0-9]+)$ ]]; then
      local lo="${BASH_REMATCH[1]}" hi="${BASH_REMATCH[2]}"
      for ((i=lo; i<=hi; i++)); do printf '%s\n' "$i"; done
    elif [[ -n "$part" ]]; then
      printf '%s\n' "$part"
    fi
  done
}

# Join PREAMBLE into a single string: cmd1 && cmd2 && ...
preamble_cmd=""
if declare -p PREAMBLE >/dev/null 2>&1; then
  # shellcheck disable=SC2128
  preamble_cmd=$(IFS=' && '; echo ${PREAMBLE[*]+"${PREAMBLE[*]}"})
fi

# --- Discover CPU pool --------------------------------------------------------

cores_list=""
if [[ -n "${CORES-}" ]]; then
  cores_list="$CORES"
else
  # Fallback to all online cores
  ncpu=$(getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 1)
  cores_list="0-$((ncpu-1))"
fi

mapfile -t ALL_CORES < <(expand_cpulist "$cores_list")
total_cores=${#ALL_CORES[@]}
(( total_cores >= k_per_job )) || { echo "CPU pool has $total_cores cores; need at least K=$k_per_job" >&2; exit 1; }

# Max parallel slots = floor(total_cores / K). At least 1.
slots=$(( total_cores / k_per_job ))
(( slots >= 1 )) || { echo "K=$k_per_job too large for pool size $total_cores" >&2; exit 1; }

# Build disjoint core groups per parallel slot: one line per slot in groups_file.
# Slot i gets cores [i*K : (i+1)*K - 1] in a simple contiguous partition.
mkdir -p "$out_dir"
: > "$groups_file"
for ((s=0; s<slots; s++)); do
  start=$(( s * k_per_job ))
  end=$(( start + k_per_job - 1 ))
  (( end < total_cores )) || break
  group=()
  for ((i=start; i<=end; i++)); do group+=("${ALL_CORES[$i]}"); done
  (IFS=,; echo "${group[*]}") >> "$groups_file"
done
# Recompute slots in case the last chunk was partial
slots=$(wc -l < "$groups_file")
(( slots >= 1 )) || { echo "No valid core groups derived." >&2; exit 1; }

# --- Filter commands: drop blanks and comments --------------------------------
# We feed cleaned commands to parallel via process substitution.
cleaned_cmds() {
  sed -e 's/^[[:space:]]*//;s/[[:space:]]*$//' \
      -e '/^$/d' \
      -e '/^#/d' "$commands_file"
}

# --- Print plan ---------------------------------------------------------------
echo "Node: $NODE"
echo "GNU parallel slots: $slots   (K per job: $k_per_job)"
echo "CPU pool: {$cores_list}"
echo "Commands: $(cleaned_cmds | wc -l | awk '{print $1}')"
echo "Results:  $results_dir"
echo "Manifest: $manifest_tsv"
echo

# --- Build the per-job runner -------------------------------------------------
# Each job:
#   1) Picks its disjoint core set from groups_file by PARALLEL_SLOT (1..slots)
#   2) Runs optional PREAMBLE
#   3) Executes the command
#
# On Linux with 'taskset', we bind to those cores. On macOS, we skip binding.
runner_linux='
CORES=$(sed -n "${PARALLEL_SLOT}p" '"$groups_file"');
if [[ -n "'"$preamble_cmd"'" ]]; then
  taskset -c "$CORES" bash -lc '"$(printf '%q' "$preamble_cmd")"' "&&" '"'"'{}'"'"'
else
  taskset -c "$CORES" bash -lc '"'"'{}'"'"'
fi
'

runner_darwin='
# taskset not available; just run.
if [[ -n "'"$preamble_cmd"'" ]]; then
  bash -lc '"$(printf '%q' "$preamble_cmd")"' "&&" '"'"'{}'"'"'
else
  bash -lc '"'"'{}'"'"'
fi
'

runner="$runner_linux"
[[ "$OS" == "Darwin" ]] && runner="$runner_darwin"
command -v taskset >/dev/null 2>&1 || [[ "$OS" == "Darwin" ]] || {
  echo "Warning: 'taskset' not found. CPU binding will be skipped." >&2
  runner="$runner_darwin"
}

# --- Run ----------------------------------------------------------------------
# Key flags:
#   --jobs "$slots"        : concurrency = number of disjoint core groups
#   --joblog manifest.tsv  : per-job TSV log (host, start/end, exit code, cmd)
#   --results results/     : captures stdout/stderr per job in a clean layout
#   --tagstring "$prefix"  : prefix tag in live output (does not affect files)
#   --lb                   : line-buffered output (less interleaving)
#   ${PARALLEL_OPTS:-}     : user-supplied extras, e.g., --bar, --halt now,fail=1
mkdir -p "$results_dir"
parallel \
  --jobs "$slots" \
  --joblog "$manifest_tsv" \
  --results "$results_dir" \
  --tagstring "$prefix" \
  --lb \
  ${PARALLEL_OPTS:-} \
  "$runner" \
  :::: <(cleaned_cmds)

echo "Done. See $manifest_tsv and $results_dir/"