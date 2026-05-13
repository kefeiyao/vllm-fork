#!/bin/bash
# Generate VLLM_NUMA_CONTROL string from ZE_AFFINITY_MASK device list.
#
# Usage:
#   ./gen_numa_control.sh <group1> [<group2> ...]
#
# Examples:
#   ./gen_numa_control.sh 0,1,2,3              # 1 group, all CPUs for these devices
#   ./gen_numa_control.sh 0,1,2,3 4,5,6,7      # 2 groups, exclusive CPU affinity
#   ./gen_numa_control.sh 0,1 2,3 4,5           # 3 groups, exclusive CPU affinity
#
# Output: VLLM_NUMA_CONTROL value(s) with CPUs evenly partitioned per worker.
# When multiple groups are given, CPUs are partitioned exclusively (no overlap).

set -e

if [[ -z "$1" ]]; then
    echo "Usage: $0 <group1> [<group2> ...]"
    echo ""
    echo "Examples:"
    echo "  $0 0,1,2,3            # 1 group"
    echo "  $0 0,1,2,3 4,5,6,7    # 2 groups, exclusive CPUs"
    echo "  $0 0,1 2,3 4,5        # 3 groups, exclusive CPUs"
    exit 1
fi

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

# All positional args are device groups
DEV_GROUPS=("$@")
MULTI_GROUP=false
if [[ ${#DEV_GROUPS[@]} -gt 1 ]]; then
    MULTI_GROUP=true
fi

# ---- Multi-group mode ----
if $MULTI_GROUP; then
    # Collect all devices across all groups and their NUMA nodes
    # Then partition each NUMA node's CPUs proportionally across groups,
    # ensuring no overlap.

    # Count devices per NUMA node per group, and total per NUMA node
    declare -A TOTAL_NODE_DEVCOUNT=()
    declare -A GROUP_NODE_DEVCOUNT=()

    for gi in "${!DEV_GROUPS[@]}"; do
        IFS=',' read -ra devs <<< "${DEV_GROUPS[$gi]}"
        for dev_id in "${devs[@]}"; do
            if [[ "$dev_id" -ge ${#XE_CARDS[@]} ]]; then
                echo "Error: device $dev_id out of range (only ${#XE_CARDS[@]} xe devices found)" >&2
                exit 1
            fi
            card="${XE_CARDS[$dev_id]}"
            numa=$(cat "$card/device/numa_node" 2>/dev/null)
            # Track per-group-per-node count using flat key "gi:numa"
            key="${gi}:${numa}"
            GROUP_NODE_DEVCOUNT[$key]=$(( ${GROUP_NODE_DEVCOUNT[$key]:-0} + 1 ))
            TOTAL_NODE_DEVCOUNT[$numa]=$(( ${TOTAL_NODE_DEVCOUNT[$numa]:-0} + 1 ))
        done
    done

    # For each NUMA node, partition CPUs across groups proportionally,
    # then within each group partition across its devices.
    # Track how many CPUs have been allocated from each node so far.
    declare -A NODE_CPU_OFFSET=()

    echo "=== Multi-group NUMA assignment ==="
    echo ""

    for gi in "${!DEV_GROUPS[@]}"; do
        IFS=',' read -ra devs <<< "${DEV_GROUPS[$gi]}"
        num_devs=${#devs[@]}
        echo "Group $gi: devices [${DEV_GROUPS[$gi]}]"

        # Collect NUMA nodes this group touches, and per-node device counts
        declare -A THIS_GROUP_NODES=()
        declare -A DEV_NUMA=()
        for dev_id in "${devs[@]}"; do
            card="${XE_CARDS[$dev_id]}"
            numa=$(cat "$card/device/numa_node" 2>/dev/null)
            DEV_NUMA[$dev_id]=$numa
            THIS_GROUP_NODES[$numa]=1
        done

        # For each NUMA node this group uses, allocate a proportional CPU slice
        declare -A GROUP_NODE_CPUS=()  # node -> space-separated CPU list for this group
        for numa in "${!THIS_GROUP_NODES[@]}"; do
            all_cpus_str=$(parse_cpulist "${NUMA_CPUS_RAW[$numa]}")
            read -ra all_cpus <<< "$all_cpus_str"
            num_cpus=${#all_cpus[@]}

            total_devs_on_node=${TOTAL_NODE_DEVCOUNT[$numa]}
            key="${gi}:${numa}"
            group_devs_on_node=${GROUP_NODE_DEVCOUNT[$key]:-0}

            # Proportional allocation: this group gets (group_devs/total_devs) of node's CPUs
            offset=${NODE_CPU_OFFSET[$numa]:-0}
            slice_size=$(( (num_cpus * group_devs_on_node) / total_devs_on_node ))
            # Last group on this node gets the remainder
            remaining_groups=0
            for gj in "${!DEV_GROUPS[@]}"; do
                kj="${gj}:${numa}"
                if [[ $gj -gt $gi && ${GROUP_NODE_DEVCOUNT[$kj]:-0} -gt 0 ]]; then
                    remaining_groups=1
                    break
                fi
            done
            if [[ $remaining_groups -eq 0 ]]; then
                slice_size=$((num_cpus - offset))
            fi

            GROUP_NODE_CPUS[$numa]="${all_cpus[@]:$offset:$slice_size}"
            NODE_CPU_OFFSET[$numa]=$((offset + slice_size))
        done

        # Now partition each node's group-slice among the group's devices on that node
        declare -A NODE_WORKER_OFFSET=()
        ENTRIES=()
        for dev_id in "${devs[@]}"; do
            numa=${DEV_NUMA[$dev_id]}
            read -ra node_cpus <<< "${GROUP_NODE_CPUS[$numa]}"
            num_node_cpus=${#node_cpus[@]}

            key="${gi}:${numa}"
            group_devs_on_node=${GROUP_NODE_DEVCOUNT[$key]}
            worker_idx=${NODE_WORKER_OFFSET[$numa]:-0}

            per_worker=$((num_node_cpus / group_devs_on_node))
            start_idx=$((worker_idx * per_worker))
            end_idx=$(( (worker_idx + 1) * per_worker - 1 ))
            if [[ $((worker_idx + 1)) -eq $group_devs_on_node ]]; then
                end_idx=$((num_node_cpus - 1))
            fi

            worker_cpus=("${node_cpus[@]:$start_idx:$((end_idx - start_idx + 1))}")
            cpu_range=$(compress_cpulist "${worker_cpus[@]}")
            ENTRIES+=("${cpu_range}:${numa}")

            NODE_WORKER_OFFSET[$numa]=$((worker_idx + 1))
        done

        RESULT=$(IFS=';'; echo "${ENTRIES[*]}")
        echo "  Per-worker assignment:"
        for i in "${!devs[@]}"; do
            echo "    local_rank $i (XPU ${devs[$i]}): ${ENTRIES[$i]}"
        done
        echo ""
        echo "  VLLM_NUMA_CONTROL=\"$RESULT\""
        echo ""

        unset THIS_GROUP_NODES DEV_NUMA GROUP_NODE_CPUS NODE_WORKER_OFFSET
    done

    exit 0
fi

# ---- Single-group mode (original behavior) ----
IFS=',' read -ra DEVICES <<< "$1"
NUM_DEVICES=${#DEVICES[@]}

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
