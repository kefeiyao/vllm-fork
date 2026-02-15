# Deepseek P/D Disaggregated Inference Framework Cheatsheet for Gaudi3 Rackscale

**Caveat:** This document applies to the current implementation of the P/D disaggregated framework, the profiling system, and the vllm version. It is subject to change as the vllm version and P/D implementation progress.

## 1. Run

### Configuration (`env_xxx.sh`)
The cluster configuration is defined in `env_xxx.sh` files (e.g., `env_JF1_8Px2_16Dx3.sh`). You must create or modify these files to define your custom cluster.

**Key Attributes:**

*   **Cluster Definition:**
    *   `ROLE_HOST[Px]`/`ROLE_IP[Px]`: Hostname and IP for Prefill nodes.
    *   `ROLE_HOST[Dx]`/`ROLE_IP[Dx]`: Hostname and IP for Decode nodes.
    *   `USR_P_NUM_INSTANCE`: Number of Prefill instances.
    *   `USR_D_NUM_INSTANCE`: Number of Decode instances.

    **Example (`env_JF1_8Px2_16Dx3.sh`):**
    *   **Prefill:** Mentions `USR_P_NUM_INSTANCE=2` with 4 nodes defined (`P0`..`P3`). This results in 2 nodes per instance (4 total nodes / 2 instances). With `USR_CARDS_PER_NODE=4`, each instance uses 8 cards (2 nodes * 4 cards). This matches `USR_PREFILL_TP_SIZE=8`.
    *   **Decode:** Mentions `USR_D_NUM_INSTANCE=3` with 12 nodes defined (`D0`..`D11`). This results in 4 nodes per instance (12 total nodes / 3 instances). With `USR_CARDS_PER_NODE=4`, each instance uses 16 cards. This matches `USR_DECODE_EP_SIZE=16`.

*   **Workload Tuning (`dp_d_env.sh` & `dp_p_env.sh`):**
    You need to adjust `max_num_seqs`, `input_min`, `input_max`, and `output_max` in these files based on your benchmark requirements.
    *   **`max_num_seqs`**: Maximum concurrency.
    *   **`input_min`/`input_max`/`output_max`**: Sequence length constraints.
    *   **Note:** Setting these values larger than necessary will slow down the warmup process.

*   **Host to NIC Mapping (`host_cx7_map.sh`):**
    For each cluster configuration, you must verify or update this file to ensure the mapping between hostnames and ConnectX-7 (CX7) device names is correct.
    *   **Function:** Maps hostnames to specific `mlx5_x` devices.
    *   **Requirement:** Ensure every node in your cluster is listed with its correct corresponding NIC identifiers.

*   **`run_on_nodes.sh`:**
    A utility script to execute commands across cluster nodes.
    *   Usage: `./run_on_nodes.sh [-p] <env_file> <command>`
    *   Use `-p` for parallel execution (experimental).

### Execution (`XPYD.sh`)
The entry point for running the framework is `XPYD.sh`.

1.  **Prerequisite:** All nodes must have **SSH password-less connection** enabled.
2.  **Invocation:** The script is invoked from the main node and uses SSH to start processes on all other nodes.
    ```bash
    ./XPYD.sh <env_file>
    # Example: ./XPYD.sh env_JF1_8Px2_16Dx3.sh
    ```
3.  **Logging:**
    *   **Local Logs:** Generated on each node under `/workspace/pd_test_log/`.
    *   **Centralized Logs:** Logs from all nodes are synced to a generic NFS folder (defined by `NFS_LOG_DIR`, default as `/host/root/kf/vllm-fork/pd_xpyd/pd_test_log`) to facilitate monitoring from the main node.
    *   **Success Indicator:** Look for "Application startup complete" in log file of each instance
        ```text
        INFO ... launcher.py:31] Route: /invocations, Methods: POST
        INFO ... launcher.py:31] Route: /start_profile, Methods: POST
        INFO ... launcher.py:31] Route: /stop_profile, Methods: POST
        INFO:     Started server process [676932]
        INFO:     Waiting for application startup.
        INFO:     Application startup complete.
        ```

## 2. Benchmark

### End-to-End Benchmark (`benchmarks/benchmark_e2e.all.sh`)
This is the key script for running end-to-end benchmarks.

1.  **Configuration:** Edit the `cases` array in `benchmarks/benchmark_e2e.all.sh` to define your test cases.
    Values are space-separated: `<input_len> <max_concurrency> <request_rate> <output_len>`
    ```bash
    cases=(
        "3500 128 inf 1000"  # Example: In=3500, Conc=128, Rate=inf, Out=1000
    )
    ```

2.  **Execution:** Run the script (recommend redirecting output to a log file).
    **Command-line Arguments:**
    *   `-m <model_path>`: Path to the model (default: `/host/mnt/kefei/HF_Models/DeepSeek-R1-Gaudi3/`).
    *   `-i <host_ip>`: Host IP address (default: `localhost`).
    *   `-e <env_file>`: Environment file (default: `./env_2p4d_sedv+.sh`).
    *   `-r <repeat_times>`: Number of times to repeat each case.

    ```bash
    # Example workflow
    ./benchmarks/benchmark_e2e.all.sh -e env_JF1_8Px2_16Dx3.sh 2>&1 | tee benchmark.log
    ```

3.  **Report:** Use `benchmarks/extract_e2e_csv.py` to extract results from the log file into a CSV format.

## 3. Profile

### Enabling & Configuring
1.  **Enable Profile:** Set `DEBUG_PROFILE=1` in `pd_env.sh`.
2.  **Trigger Profile:** Use the helper script `generate_profile_config.sh`.
    *   This script creates a configuration file that triggers profiling when specific conditions are met (e.g., number of inflight requests).
    *   **Usage:**
        ```bash
        ./generate_profile_config.sh -i <inflight> -b <block_size> -s <steps> -r <ranks> -t <target> -p <config_dir>
        ```
    *   Execution of this script effectively "arms" the system to profile once the workload matches the config.

### Analysis
1.  **Output:** The profile file will be generated under `WORK_DIR` once the profiling finishes.
2.  **Visualization:** Open and analyze the generated profile using [https://perfetto.habana.ai/](https://perfetto.habana.ai/).
