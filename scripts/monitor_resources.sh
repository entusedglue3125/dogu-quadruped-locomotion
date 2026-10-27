#!/usr/bin/env bash
# ==============================================================================
# DOGU Locomotion Resource Monitoring Script
# Continuously logs and monitors GPU VRAM, System RAM, Swap, and CPU load.
# ==============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_FILE="$PROJECT_ROOT/logs/resource_usage.csv"
INTERVAL=${1:-2} # sample every 2 seconds by default

mkdir -p "$(dirname "$LOG_FILE")"

# Write CSV header if file doesn't exist
if [ ! -f "$LOG_FILE" ]; then
    echo "timestamp,gpu_util_pct,gpu_vram_used_mb,gpu_vram_free_mb,gpu_temp_c,ram_used_mb,ram_avail_mb,swap_used_mb" > "$LOG_FILE"
fi

echo "=============================================================================="
echo "          DOGU Quadruped Locomotion - System Resource Monitor                 "
echo "=============================================================================="
echo "[*] Sampling every ${INTERVAL}s. Logging to: $LOG_FILE"
echo "[*] Press Ctrl+C to stop."
echo "------------------------------------------------------------------------------"
printf "%-19s | %-8s | %-13s | %-6s | %-12s | %-9s\n" "Timestamp" "GPU Util" "VRAM (Used/Tot)" "Temp" "RAM (Used/Tot)" "Swap Used"
echo "------------------------------------------------------------------------------"

while true; do
    TS=$(date +"%Y-%m-%d %H:%M:%S")

    # GPU stats via nvidia-smi
    GPU_DATA=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.free,temperature.gpu --format=csv,noheader,nounits 2>/dev/null || echo "0, 0, 0, 0")
    GPU_UTIL=$(echo "$GPU_DATA" | awk -F',' '{print $1}' | tr -d ' ')
    VRAM_USED=$(echo "$GPU_DATA" | awk -F',' '{print $2}' | tr -d ' ')
    VRAM_FREE=$(echo "$GPU_DATA" | awk -F',' '{print $3}' | tr -d ' ')
    GPU_TEMP=$(echo "$GPU_DATA" | awk -F',' '{print $4}' | tr -d ' ')

    # Host RAM stats via free -m
    RAM_TOTAL=$(free -m | awk '/^Mem:/ {print $2}')
    RAM_USED=$(free -m | awk '/^Mem:/ {print $3}')
    RAM_AVAIL=$(free -m | awk '/^Mem:/ {print $7}')
    SWAP_USED=$(free -m | awk '/^Swap:/ {print $3}')

    # Append to CSV
    echo "${TS},${GPU_UTIL},${VRAM_USED},${VRAM_FREE},${GPU_TEMP},${RAM_USED},${RAM_AVAIL},${SWAP_USED}" >> "$LOG_FILE"

    # Display row
    printf "%-19s | %3s%%     | %5s / 6144 MB | %3s°C  | %5s / %5s MB | %5s MB\n" \
           "$TS" "$GPU_UTIL" "$VRAM_USED" "$GPU_TEMP" "$RAM_USED" "$RAM_TOTAL" "$SWAP_USED"

    sleep "$INTERVAL"
done
