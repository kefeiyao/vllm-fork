#!/bin/bash
# Generate VLLM_NUMA_CONTROL string from ZE_AFFINITY_MASK device list.
#
# Usage: ./gen_numa_control.sh <device_list>
#   e.g.: ./gen_numa_control.sh 3,4,5,6
#
# Output: VLLM_NUMA_CONTROL value with CPUs evenly partitioned per worker.

set -e

if [[ -z "$1" ]]; then
    echo "Usage: $0 <device_list>"
    echo "  e.g.: $0 3,4,5,6"
    exit 1
fi

IFS=',' read -ra DEVICES <<< "$1"
NUM_DEVICES=${#DEVICES[@]}

# Discover xe-driver DRM cards (skip non-xe like AST/BMC)
declare -a XE_CARDS=()
for card in /sys/class/drm/card[0-9]*; do
    [[ -d "$card" ]] || continue
    # skip sub-devices like card1-DP-1
    basename "$card" | grep -qP '^card\d+$' || continue
    driver=$(basename "$(readlink -f "$card/device/driver" 2>/dev/null)" 2>/dev/null)
    if [[ "$driver" == "xe" ]]; then
        XE_CARDS+=("$card")
    fi
done

if [[ ${#XE_CARDS[@]} -eq 0 ]]; then
    echo "Error: no xe-driver GPU cards found in /sys/class/drm/" >&2
    exit 1
fi

# Sort xe cards by card number
IFS=$'\n' XE_CARDS=($(printf '%s\n' "${XE_CARDS[@]}" | sort -V)); unset IFS

echo "Detected ${#XE_CARDS[@]} xe devices:"
for i in "${!XE_CARDS[@]}"; do
    card="${XE_CARDS[$i]}"
    numa=$(cat "$card/device/numa_node" 2>/dev/null)
    echo "  ZE device $i -> $(basename "$card"), NUMA node $numa"
done
echo ""

# Collect NUMA node CPUs
declare -A NUMA_CPUS_RAW=()
for node_dir in /sys/devices/system/node/node[0-9]*; do
    [[ -d "$node_dir" ]] || continue
    node_id=$(basename "$node_dir" | sed 's/node//')
    cpulist=$(cat "$node_dir/cpulist" 2>/dev/null)
    NUMA_CPUS_RAW[$node_id]="$cpulist"
done

# Parse a cpulist string into an array of CPU IDs
parse_cpulist() {
    local cpulist="$1"
    local -a cpus=()
    IFS=',' read -ra parts <<< "$cpulist"
    for part in "${parts[@]}"; do
        if [[ "$part" == *-* ]]; then
            lo="${part%-*}"
            hi="${part#*-}"
            for ((c=lo; c<=hi; c++)); do
                cpus+=("$c")
            done
        else
            cpus+=("$part")
        fi
    done
    echo "${cpus[@]}"
}

# Count how many of the requested devices land on each NUMA node
declare -A NODE_DEVICE_COUNT=()
declare -A DEVICE_NUMA=()
for dev_id in "${DEVICES[@]}"; do
    if [[ "$dev_id" -ge ${#XE_CARDS[@]} ]]; then
        echo "Error: device $dev_id out of range (only ${#XE_CARDS[@]} xe devices found)" >&2
        exit 1
    fi
    card="${XE_CARDS[$dev_id]}"
    numa=$(cat "$card/device/numa_node" 2>/dev/null)
    DEVICE_NUMA[$dev_id]=$numa
    NODE_DEVICE_COUNT[$numa]=$(( ${NODE_DEVICE_COUNT[$numa]:-0} + 1 ))
done

# For each NUMA node, partition its CPUs among the workers on that node
declare -A NODE_ASSIGNED=()  # track how many workers assigned per node so far
ENTRIES=()

for dev_id in "${DEVICES[@]}"; do
    numa=${DEVICE_NUMA[$dev_id]}
    total_on_node=${NODE_DEVICE_COUNT[$numa]}
    assigned_so_far=${NODE_ASSIGNED[$numa]:-0}

    # Get all CPUs for this NUMA node
    all_cpus_str=$(parse_cpulist "${NUMA_CPUS_RAW[$numa]}")
    read -ra all_cpus <<< "$all_cpus_str"
    num_cpus=${#all_cpus[@]}

    # Partition evenly
    per_worker=$((num_cpus / total_on_node))
    start_idx=$((assigned_so_far * per_worker))
    end_idx=$(( (assigned_so_far + 1) * per_worker - 1 ))
    # Last worker gets remainder
    if [[ $((assigned_so_far + 1)) -eq $total_on_node ]]; then
        end_idx=$((num_cpus - 1))
    fi

    # Extract this worker's CPU slice
    worker_cpus=("${all_cpus[@]:$start_idx:$((end_idx - start_idx + 1))}")

    # Compress CPU list into ranges for readability
    compress_cpulist() {
        local -a sorted=($(printf '%s\n' "$@" | sort -n))
        local result=""
        local range_start=${sorted[0]}
        local range_end=${sorted[0]}
        for ((i=1; i<${#sorted[@]}; i++)); do
            if [[ ${sorted[$i]} -eq $((range_end + 1)) ]]; then
                range_end=${sorted[$i]}
            else
                if [[ $range_start -eq $range_end ]]; then
                    result+="${range_start},"
                else
                    result+="${range_start}-${range_end},"
                fi
                range_start=${sorted[$i]}
                range_end=${sorted[$i]}
            fi
        done
        if [[ $range_start -eq $range_end ]]; then
            result+="${range_start}"
        else
            result+="${range_start}-${range_end}"
        fi
        echo "$result"
    }

    cpu_range=$(compress_cpulist "${worker_cpus[@]}")
    ENTRIES+=("${cpu_range}:${numa}")

    NODE_ASSIGNED[$numa]=$((assigned_so_far + 1))
done

# Join entries with semicolons
RESULT=$(IFS=';'; echo "${ENTRIES[*]}")

echo "Device list: $1"
echo ""
echo "Per-worker assignment:"
for i in "${!DEVICES[@]}"; do
    echo "  local_rank $i (XPU ${DEVICES[$i]}): ${ENTRIES[$i]}"
done
echo ""
echo "VLLM_NUMA_CONTROL=\"$RESULT\""
