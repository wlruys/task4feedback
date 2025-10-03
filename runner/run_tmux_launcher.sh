#!/usr/bin/env bash
# run_tmux_launcher.sh
# -----------------------------------------------------------------------------
# Launch each command in its own tmux session, allocating K unique CPU cores
# per session (disjoint concurrently on this node). When a session finishes,
# its cores are freed and reused. Multiple nodes may run this concurrently and
# safely append to a shared manifest.csv via mkdir-based lock.
#
# Usage:
#   ./run_tmux_launcher.sh <commands.txt> <K> [prefix] [log_dir]
#
# Environment overrides:
#   CORES="0-15,32-47"           # restrict the core pool (Linux/macOS)
#   PREAMBLE=("cmd1" "cmd2")     # array: snippets run before each command
#   TMUX_PREFIX="..."            # overrides [prefix] arg if set
#   TMUX_LOG_DIR="..."           # overrides [log_dir] arg if set
# -----------------------------------------------------------------------------

set -euo pipefail

commands_file="${1:-commands.txt}"
k_per_session="${2:-1}"
prefix_arg="${3:-mysess}"
log_dir_arg="${4:-logs/tmux_sessions}"

# Allow SLURM/exported overrides:
prefix="${TMUX_PREFIX:-$prefix_arg}"
log_dir="${TMUX_LOG_DIR:-$log_dir_arg}"

[[ -f "$commands_file" ]] || { echo "Missing commands file: $commands_file" >&2; exit 1; }
[[ "$k_per_session" =~ ^[0-9]+$ && "$k_per_session" -ge 1 ]] || { echo "K must be a positive integer" >&2; exit 1; }
command -v tmux >/dev/null 2>&1 || { echo "tmux not found" >&2; exit 1; }

mkdir -p "$log_dir"
state_dir="$log_dir/.state"
mkdir -p "$state_dir"

OS=$(uname -s || echo "Unknown")
NODE=$(hostname -s 2>/dev/null || hostname || echo "unknown-node")
PID=$$

now_utc() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

# Single-quote for safe shell embedding
sq() { local s=${1//\'/\'\\\'\'}; printf "'%s'" "$s"; }

# Expand "0-3,5,7-9" -> one per line
expand_cpulist() {
  local list="${1:-}" IFS=',' part
  set +u
  for part in $list; do
    if [[ "$part" =~ ^([0-9]+)-([0-9]+)$ ]]; then
      local lo="${BASH_REMATCH[1]}" hi="${BASH_REMATCH[2]}" i
      for ((i=lo; i<=hi; i++)); do printf '%s\n' "$i"; done
    elif [[ -n "$part" ]]; then
      printf '%s\n' "$part"
    fi
  done
  set -u
}

csv_escape() { local s="$1"; s=${s//\"/\"\"}; printf '"%s"' "$s"; }

# mkdir-based lock (portable on Lustre/NFS/ext4)
with_lock() {
  local lockdir="$1"; shift
  local waits=(0.02 0.05 0.1 0.2 0.4 0.8 1.0 1.0 1.0 1.0) # capped backoff
  local t_start=$(date +%s) timeout=60
  while ! mkdir "$lockdir" 2>/dev/null; do
    if [[ -e "$lockdir" ]]; then
      local mt now
      mt=$(stat -c %Y "$lockdir" 2>/dev/null || stat -f %m "$lockdir" 2>/dev/null || echo 0)
      now=$(date +%s)
      if (( now - mt > 600 )); then rmdir "$lockdir" 2>/dev/null || true; fi
    fi
    local w=${waits[0]}; waits=("${waits[@]:1}" "$w")
    sleep "$w"
    if (( $(date +%s) - t_start > timeout )); then
      echo "Lock timeout: $lockdir" >&2
      return 1
    fi
  done
  local rc=0
  ( set +e; "$@" ); rc=$?
  rmdir "$lockdir" 2>/dev/null || true
  return $rc
}

append_manifest() {
  local line="$1"
  local manifest="$2"
  local lockdir="${manifest}.lockdir"
  with_lock "$lockdir" bash -c '
    set -e
    mf=$1; ln=$2
    if [[ ! -s "$mf" ]]; then
      printf "%s\n" "node,session,cores,command,stdout,stderr,start_time,pid" > "$mf"
    fi
    printf "%s\n" "$ln" >> "$mf"
  ' bash "$manifest" "$line"
}

# -------------------- Discover core pool --------------------
cores_list=""
if [[ -n "${CORES-}" ]]; then
  cores_list="$CORES"
else
  if [[ "$OS" == "Linux" ]] && command -v taskset >/dev/null 2>&1; then
    cores_list="$(taskset -pc $$ 2>/dev/null | awk -F': ' '{print $2}')"
  fi
  if [[ -z "$cores_list" ]]; then
    ncpu=$(getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 1)
    cores_list="0-$((ncpu-1))"
  fi
fi

FREE_CORES=()
while IFS= read -r c; do
  [[ -n "$c" ]] && FREE_CORES+=("$c")
done < <(expand_cpulist "$cores_list")

total_cores=${#FREE_CORES[@]}
(( total_cores >= k_per_session )) || { echo "CPU pool has $total_cores cores; need at least K=$k_per_session" >&2; exit 1; }
max_parallel=$(( total_cores / k_per_session ))
(( max_parallel >= 1 )) || { echo "K=$k_per_session too large for pool size $total_cores" >&2; exit 1; }

# -------------------- Load commands (skip blanks/#) --------------------
COMMANDS=()
while IFS= read -r line || [[ -n "$line" ]]; do
  line="${line#"${line%%[![:space:]]*}"}"
  line="${line%"${line##*[![:space:]]}"}"
  [[ -z "$line" || "$line" =~ ^# ]] && continue
  COMMANDS+=("$line")
done < "$commands_file"

total_cmds=${#COMMANDS[@]}
(( total_cmds > 0 )) || { echo "No runnable commands in $commands_file" >&2; exit 1; }

# Optional preamble from env array PREAMBLE
preamble_cmd=""
if declare -p PREAMBLE >/dev/null 2>&1; then
  # shellcheck disable=SC2128
  preamble_cmd=$(IFS=' && '; echo ${PREAMBLE[*]+"${PREAMBLE[*]}"})
fi

# --------------- Core allocator ---------------
pop_cores() {
  local __out="$1" need="$2"
  local have="${#FREE_CORES[@]}"
  (( have >= need )) || return 1
  local picked=() remaining=() i=0 c
  for c in "${FREE_CORES[@]}"; do
    if (( i < need )); then picked+=("$c"); else remaining+=("$c"); fi
    ((i++))
  done
  FREE_CORES=("${remaining[@]}")
  eval "$__out=(\"\${picked[@]}\")"
  return 0
}
push_cores() { local c; for c in "$@"; do FREE_CORES+=("$c"); done; }
session_exists() { tmux has-session -t "$1" >/dev/null 2>&1; }

# Handle interrupts: kill any running sessions cleanly
terminate() {
  echo "Signal received; terminating running tmux sessions..." >&2
  for s in "${RUNNING[@]:-}"; do tmux kill-session -t "$s" 2>/dev/null || true; done
  exit 130
}
trap terminate INT TERM

# --------------- Execution wrapper per session ---------------
write_wrapper_and_launch() {
  local session="$1" core_csv="$2" full_cmd="$3" log_file="$4" err_file="$5"
  local wrapper="${state_dir}/run_${session}.sh"
  {
    echo '#!/usr/bin/env bash'
    echo 'set -euo pipefail'
    if [[ "$OS" == "Linux" ]] && command -v taskset >/dev/null 2>&1; then
      printf 'exec bash -lc %s > %s 2> %s\n' \
        "$(sq "taskset -c ${core_csv} bash -lc $(sq "$full_cmd")")" \
        "$(sq "$log_file")" \
        "$(sq "$err_file")"
    else
      printf 'exec bash -lc %s > %s 2> %s\n' \
        "$(sq "$full_cmd")" \
        "$(sq "$log_file")" \
        "$(sq "$err_file")"
    fi
  } > "$wrapper"
  chmod +x "$wrapper"
  tmux new-session -d -s "$session" "$wrapper"
}

# ---------------------------- Scheduler ----------------------------
echo "Node: $NODE | PID: $PID"
echo "CPU pool size: $total_cores (pool={$cores_list})"
echo "K per session: $k_per_session  -> max parallel sessions: $max_parallel"
echo "Total jobs: $total_cmds"
echo "Logs: $log_dir  | Manifest: $log_dir/manifest.csv"
echo "Starting..."

RUNNING=()
next_cmd=0

launch_one() {
  local idx="$1"
  local session="${prefix}_${idx}"
  local cmd="${COMMANDS[$idx]}"

  local picked_arr=()
  if ! pop_cores picked_arr "$k_per_session"; then
    return 1
  fi

  local core_csv; core_csv=$(IFS=,; echo "${picked_arr[*]}")
  local start_time; start_time=$(now_utc)
  local log_file="${log_dir}/${session}.log"
  local err_file="${log_dir}/${session}.err"

  local full_cmd
  if [[ -n "$preamble_cmd" ]]; then
    full_cmd="(${preamble_cmd}) && ${cmd}"
  else
    full_cmd="${cmd}"
  fi

  echo "Launching [$session] on cores {${core_csv}}: $cmd"

  { for c in "${picked_arr[@]}"; do echo "$c"; done; } > "${state_dir}/${session}.cores"

  write_wrapper_and_launch "$session" "$core_csv" "$full_cmd" "$log_file" "$err_file"

  local line
  line="$(csv_escape "$NODE"),$(csv_escape "$session"),$(csv_escape "{$core_csv}"),$(csv_escape "$cmd"),$(csv_escape "$log_file"),$(csv_escape "$err_file"),$(csv_escape "$start_time"),$(csv_escape "$PID")"
  append_manifest "$line" "$log_dir/manifest.csv"

  RUNNING+=("$session")
  return 0
}

while (( next_cmd < total_cmds || ${#RUNNING[@]} > 0 )); do
  while (( next_cmd < total_cmds )); do
    ((${#FREE_CORES[@]} >= k_per_session)) || break
    ((${#RUNNING[@]} < max_parallel)) || break
    launch_one "$next_cmd" && ((next_cmd++)) || break
  done

  if ((${#RUNNING[@]} > 0)); then
    NEW_RUNNING=()
    for s in "${RUNNING[@]}"; do
      if session_exists "$s"; then
        NEW_RUNNING+=("$s")
      else
        cores_to_free=()
        if [[ -f "${state_dir}/${s}.cores" ]]; then
          while IFS= read -r cc; do [[ -n "$cc" ]] && cores_to_free+=("$cc"); done < "${state_dir}/${s}.cores"
          rm -f "${state_dir}/${s}.cores" 2>/dev/null || true
        fi
        if ((${#cores_to_free[@]} > 0)); then
          push_cores "${cores_to_free[@]}"
          echo "Finished [$s], freed cores {$(IFS=,; echo "${cores_to_free[*]}")}"
        else
          echo "Finished [$s], (no core record found)"
        fi
      fi
    done
    RUNNING=("${NEW_RUNNING[@]}")
  fi

  if (( next_cmd >= total_cmds )) && ((${#RUNNING[@]} == 0)); then
    break
  fi

  sleep 0.2
done

echo "All sessions finished."