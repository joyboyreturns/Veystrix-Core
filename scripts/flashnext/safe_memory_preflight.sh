#!/usr/bin/env bash
# safe_memory_preflight.sh - Safe macOS memory preflight and cache recovery gate for Veystrix.
#
# Usage:
#   scripts/flashnext/safe_memory_preflight.sh [OPTIONS]
#
# Options:
#   --checkpoint PATH       Model checkpoint directory (default: ~/models/Qwen3.8-Flash-Next-MLX-4bit-MTP)
#   --threshold-gb VALUE    Startup threshold in GiB (default: 9.0)
#   --dry-run               Simulate memory checks and purge decisions without executing purge
#   --json                  Output structured JSON to stdout
#   --run-preflight         If memory gate passes, execute python preflight check
#   --force-purge           Attempt purge even if memory is close to threshold (obeys safety limits)
#   --state-file PATH       Custom purge state tracker file (default: /tmp/flashnext_purge_state.json)
#   -h, --help              Show this help message

set -euo pipefail

# Default configuration
DEFAULT_CHECKPOINT="${HOME}/models/Qwen3.8-Flash-Next-MLX-4bit-MTP"
CHECKPOINT="${DEFAULT_CHECKPOINT}"
THRESHOLD_GB="9.0"
DRY_RUN="0"
JSON_OUTPUT="0"
RUN_PREFLIGHT="0"
FORCE_PURGE="0"
STATE_FILE="/tmp/flashnext_purge_state.json"
MAX_PURGES=2
MIN_PURGE_INTERVAL_SECONDS=1800  # 30 minutes
SETTLE_WAIT_SECONDS=3

# Locate Python executable
PYTHON="python3"
if [[ -x "./.venv/bin/python" ]]; then
    PYTHON="./.venv/bin/python"
elif [[ -x "${HOME}/Documents/Veystrix/.venv/bin/python" ]]; then
    PYTHON="${HOME}/Documents/Veystrix/.venv/bin/python"
fi

# Locate purge command
PURGE_CMD=""
if [[ -x "/usr/sbin/purge" ]]; then
    PURGE_CMD="/usr/sbin/purge"
elif command -v purge >/dev/null 2>&1; then
    PURGE_CMD="$(command -v purge)"
fi

# Parse CLI arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --threshold-gb)
            THRESHOLD_GB="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN="1"
            shift
            ;;
        --json)
            JSON_OUTPUT="1"
            shift
            ;;
        --run-preflight)
            RUN_PREFLIGHT="1"
            shift
            ;;
        --force-purge)
            FORCE_PURGE="1"
            shift
            ;;
        --state-file)
            STATE_FILE="$2"
            shift 2
            ;;
        -h|--help)
            sed -ne '/^#/!q;s/^# //;2,$p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

# Expand tilde in checkpoint path if present
CHECKPOINT="${CHECKPOINT/#\~/$HOME}"

# Python helper to inspect memory, state, and checkpoint
READ_SCRIPT=$(cat << 'EOF'
import sys
import os
import json
import time
import subprocess
import re
from pathlib import Path

def parse_vm_stat(text: str) -> dict:
    match = re.search(r"page size of (\d+) bytes", text)
    page = int(match.group(1)) if match else 16384
    pages, counters = {}, {}
    names = {
        "Pages free": "free", "Pages inactive": "inactive", "Pages speculative": "speculative",
        "Pages purgeable": "purgeable", "Pages occupied by compressor": "compressor",
        "Pages wired down": "wired", "Pages active": "active",
        "File-backed pages": "file_backed", "Anonymous pages": "anonymous"
    }
    wanted = {
        "Swapins": "swapin", "Swapouts": "swapout", "Pageins": "pagein", "Pageouts": "pageout",
        "Compressions": "compress", "Decompressions": "decompress"
    }
    for line in text.splitlines():
        label, _, value = line.partition(":")
        label, value = label.strip().strip('"'), value.strip().rstrip(".")
        if not value.isdigit():
            continue
        if label in names:
            pages[names[label]] = int(value)
        elif label in wanted:
            counters[wanted[label]] = int(value)
            
    mb = lambda k: pages.get(k, 0) * page / 1048576
    available = mb("free") + mb("inactive") + mb("speculative")
    return {
        "page_bytes": page,
        "available_mb": round(available, 1),
        "available_gb": (pages.get("free", 0) + pages.get("inactive", 0) + pages.get("speculative", 0)) * page / 1e9,
        "free_mb": round(mb("free"), 1),
        "inactive_mb": round(mb("inactive"), 1),
        "speculative_mb": round(mb("speculative"), 1),
        "purgeable_mb": round(mb("purgeable"), 1),
        "compressor_mb": round(mb("compressor"), 1),
        "wired_mb": round(mb("wired"), 1),
        "active_mb": round(mb("active"), 1),
        "file_backed_mb": round(mb("file_backed"), 1),
        "anonymous_mb": round(mb("anonymous"), 1),
        "counters": counters
    }

def read_system():
    try:
        vm_out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5).stdout
        mem = parse_vm_stat(vm_out)
    except Exception as e:
        mem = {"error": str(e), "available_gb": 0.0, "available_mb": 0.0}

    swap_used_mb = 0.0
    try:
        swap_out = subprocess.run(["sysctl", "vm.swapusage"], capture_output=True, text=True, timeout=5).stdout
        m = re.search(r"used = ([\d.]+)M", swap_out)
        if m:
            swap_used_mb = float(m.group(1))
    except Exception:
        pass
    mem["swap_used_mb"] = swap_used_mb

    pressure_level = 1
    try:
        press_out = subprocess.run(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"],
                                   capture_output=True, text=True, timeout=5).stdout.strip()
        if press_out.isdigit():
            pressure_level = int(press_out)
    except Exception:
        pass
    mem["pressure_level"] = pressure_level
    return mem

def check_checkpoint(ckpt_path: str):
    p = Path(ckpt_path)
    if not p.exists():
        return False, f"Checkpoint directory does not exist: {ckpt_path}"
    if not p.is_dir():
        return False, f"Checkpoint path is not a directory: {ckpt_path}"
    # Check for config.json or weights
    config_file = p / "config.json"
    if not config_file.exists():
        return False, f"Missing config.json in checkpoint: {ckpt_path}"
    return True, "Checkpoint valid"

def check_competing_workloads():
    # Detect running heavy inference or benchmark processes
    try:
        out = subprocess.run(["ps", "-eo", "pid,command"], capture_output=True, text=True, timeout=5).stdout
        competing = []
        my_pid = os.getpid()
        parent_pid = os.getppid()
        for line in out.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) < 2:
                continue
            pid_str, cmd = parts[0], parts[1]
            if not pid_str.isdigit():
                continue
            pid = int(pid_str)
            if pid in (my_pid, parent_pid):
                continue
            if "bench_prefill_memory" in cmd and "--child" in cmd:
                competing.append({"pid": pid, "cmd": cmd})
            elif "veystrix" in cmd and "generate" in cmd:
                competing.append({"pid": pid, "cmd": cmd})
        return competing
    except Exception:
        return []

def read_purge_state(state_file: str):
    p = Path(state_file)
    if not p.exists():
        return {"count": 0, "last_purge_timestamp": 0, "history": []}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {"count": 0, "last_purge_timestamp": 0, "history": []}

action = sys.argv[1]
if action == "sample":
    print(json.dumps(read_system()))
elif action == "checkpoint":
    valid, msg = check_checkpoint(sys.argv[2])
    print(json.dumps({"valid": valid, "message": msg}))
elif action == "competing":
    print(json.dumps(check_competing_workloads()))
elif action == "state":
    print(json.dumps(read_purge_state(sys.argv[2])))
EOF
)

# Step 1: Check competing heavy workloads
COMPETING_JSON="$("${PYTHON}" -c "${READ_SCRIPT}" competing)"
HAS_COMPETING="$("${PYTHON}" -c "import json, sys; c = json.loads(sys.argv[1]); print('1' if c else '0')" "${COMPETING_JSON}")"
if [[ "${HAS_COMPETING}" == "1" ]]; then
    REASON="Competing inference workload active: ${COMPETING_JSON}"
    if [[ "${JSON_OUTPUT}" == "1" ]]; then
        "${PYTHON}" -c "import json; print(json.dumps({'status': 'BLOCKED', 'code': 3, 'reason': '${REASON}'}))"
    else
        echo "ERROR: ${REASON}" >&2
    fi
    exit 3
fi

# Step 2: Check model checkpoint
CKPT_STATUS="$("${PYTHON}" -c "${READ_SCRIPT}" checkpoint "${CHECKPOINT}")"
CKPT_VALID="$("${PYTHON}" -c "import json, sys; print('1' if json.loads(sys.argv[1]).get('valid') else '0')" "${CKPT_STATUS}")"
if [[ "${CKPT_VALID}" != "1" ]]; then
    MSG="$("${PYTHON}" -c "import json, sys; print(json.loads(sys.argv[1]).get('message', ''))" "${CKPT_STATUS}")"
    REASON="Model checkpoint invalid: ${MSG}"
    if [[ "${JSON_OUTPUT}" == "1" ]]; then
        "${PYTHON}" -c "import json; print(json.dumps({'status': 'BLOCKED', 'code': 2, 'reason': '${REASON}'}))"
    else
        echo "ERROR: ${REASON}" >&2
    fi
    exit 2
fi

# Step 3: Measure initial memory & pressure
MEM_INITIAL="$("${PYTHON}" -c "${READ_SCRIPT}" sample)"
INITIAL_AVAIL_GB="$("${PYTHON}" -c "import json, sys; print(json.loads(sys.argv[1])['available_gb'])" "${MEM_INITIAL}")"
INITIAL_PRESSURE="$("${PYTHON}" -c "import json, sys; print(json.loads(sys.argv[1])['pressure_level'])" "${MEM_INITIAL}")"
INITIAL_SWAP_USED="$("${PYTHON}" -c "import json, sys; print(json.loads(sys.argv[1])['swap_used_mb'])" "${MEM_INITIAL}")"
FILE_BACKED_MB="$("${PYTHON}" -c "import json, sys; print(json.loads(sys.argv[1])['file_backed_mb'])" "${MEM_INITIAL}")"

# Step 4: Check if memory pressure is critical (>= 3) or heavy swapping
if [[ "${INITIAL_PRESSURE}" -ge 3 ]]; then
    REASON="System is under critical memory pressure (level ${INITIAL_PRESSURE})"
    if [[ "${JSON_OUTPUT}" == "1" ]]; then
        "${PYTHON}" -c "import json; print(json.dumps({'status': 'BLOCKED', 'code': 4, 'reason': '${REASON}', 'initial_memory': json.loads(sys.argv[1])}))" "${MEM_INITIAL}"
    else
        echo "ERROR: ${REASON}" >&2
    fi
    exit 4
fi

# Step 5: Check against startup threshold
MEETS_THRESHOLD="$("${PYTHON}" -c "import sys; avail=float(sys.argv[1]); thresh=float(sys.argv[2])*1.073741824; print('1' if avail >= thresh else '0')" "${INITIAL_AVAIL_GB}" "${THRESHOLD_GB}")"

PURGE_ATTEMPTED="0"
PURGE_SUCCESS="0"
PURGE_SKIPPED_REASON=""
PURGE_RECOVERED_MB="0.0"
MEM_FINAL="${MEM_INITIAL}"
FINAL_AVAIL_GB="${INITIAL_AVAIL_GB}"
FINAL_PRESSURE="${INITIAL_PRESSURE}"

if [[ "${MEETS_THRESHOLD}" == "1" && "${FORCE_PURGE}" != "1" ]]; then
    PURGE_SKIPPED_REASON="available memory ${INITIAL_AVAIL_GB} decimal GB already meets startup threshold ${THRESHOLD_GB} GiB"
elif [[ "${FORCE_PURGE}" != "1" ]]; then
    PURGE_SKIPPED_REASON="purge not requested; memory is below the startup threshold"
else
    # Memory is insufficient or --force-purge set; evaluate purge safety rules
    PURGE_STATE="$("${PYTHON}" -c "${READ_SCRIPT}" state "${STATE_FILE}")"
    PURGE_COUNT="$("${PYTHON}" -c "import json, sys; print(json.loads(sys.argv[1]).get('count', 0))" "${PURGE_STATE}")"
    LAST_PURGE_TS="$("${PYTHON}" -c "import json, sys; print(json.loads(sys.argv[1]).get('last_purge_timestamp', 0))" "${PURGE_STATE}")"
    NOW_TS="$(date +%s)"
    TIME_SINCE_LAST="$(( NOW_TS - LAST_PURGE_TS ))"

    if [[ -z "${PURGE_CMD}" ]]; then
        PURGE_SKIPPED_REASON="purge binary not found on system"
    elif [[ "${PURGE_COUNT}" -ge "${MAX_PURGES}" ]]; then
        PURGE_SKIPPED_REASON="maximum purge limit reached (${PURGE_COUNT}/${MAX_PURGES} for this mission)"
    elif [[ "${LAST_PURGE_TS}" -gt 0 && "${TIME_SINCE_LAST}" -lt "${MIN_PURGE_INTERVAL_SECONDS}" ]]; then
        SETTLE_REMAINING="$(( MIN_PURGE_INTERVAL_SECONDS - TIME_SINCE_LAST ))"
        PURGE_SKIPPED_REASON="purge settling constraint: last purge was ${TIME_SINCE_LAST}s ago (${SETTLE_REMAINING}s remaining in 30m window)"
    elif [[ "${DRY_RUN}" == "1" ]]; then
        PURGE_ATTEMPTED="1"
        PURGE_SKIPPED_REASON="dry-run mode: purge execution simulated"
    else
        # Test sudo -n without prompting for password
        PURGE_ATTEMPTED="1"
        if sudo -n "${PURGE_CMD}" >/dev/null 2>&1; then
            PURGE_SUCCESS="1"
            # Wait for memory counters to settle
            sleep "${SETTLE_WAIT_SECONDS}"
            MEM_FINAL="$("${PYTHON}" -c "${READ_SCRIPT}" sample)"
            FINAL_AVAIL_GB="$("${PYTHON}" -c "import json, sys; print(json.loads(sys.argv[1])['available_gb'])" "${MEM_FINAL}")"
            FINAL_PRESSURE="$("${PYTHON}" -c "import json, sys; print(json.loads(sys.argv[1])['pressure_level'])" "${MEM_FINAL}")"
            PURGE_RECOVERED_MB="$("${PYTHON}" -c "import json, sys; m1=json.loads(sys.argv[1]); m2=json.loads(sys.argv[2]); print(round(m2['available_mb'] - m1['available_mb'], 1))" "${MEM_INITIAL}" "${MEM_FINAL}")"
            
            # Update state file
            "${PYTHON}" -c "
import json, sys, time
p = sys.argv[1]
try:
    with open(p, 'r') as f:
        data = json.load(f)
except Exception:
    data = {'count': 0, 'last_purge_timestamp': 0, 'history': []}
data['count'] = data.get('count', 0) + 1
now = int(time.time())
data['last_purge_timestamp'] = now
data.setdefault('history', []).append({
    'timestamp': now,
    'recovered_mb': float(sys.argv[2]),
    'available_before_gb': float(sys.argv[3]),
    'available_after_gb': float(sys.argv[4])
})
with open(p, 'w') as f:
    json.dump(data, f, indent=2)
" "${STATE_FILE}" "${PURGE_RECOVERED_MB}" "${INITIAL_AVAIL_GB}" "${FINAL_AVAIL_GB}"
        else
            PURGE_SUCCESS="0"
            PURGE_SKIPPED_REASON="sudo -n /usr/sbin/purge requires password or failed (non-interactive mode preserved)"
        fi
    fi
fi

# Step 6: Final check against readiness threshold
GATE_PASSED="$("${PYTHON}" -c "import sys; avail=float(sys.argv[1]); thresh=float(sys.argv[2])*1.073741824; print('1' if avail >= thresh else '0')" "${FINAL_AVAIL_GB}" "${THRESHOLD_GB}")"

# Step 7: Formulate report
if [[ "${GATE_PASSED}" == "1" ]]; then
    STATUS="READY"
    EXIT_CODE=0
    REASON="available memory ${FINAL_AVAIL_GB} decimal GB passes startup threshold ${THRESHOLD_GB} GiB"
else
    STATUS="BLOCKED"
    EXIT_CODE=1
    REASON="available memory ${FINAL_AVAIL_GB} decimal GB is below startup threshold ${THRESHOLD_GB} GiB"
    if [[ -n "${PURGE_SKIPPED_REASON}" ]]; then
        REASON="${REASON} (${PURGE_SKIPPED_REASON})"
    fi
fi

if [[ "${JSON_OUTPUT}" == "1" ]]; then
    "${PYTHON}" -c "
import json, sys
data = {
    'status': sys.argv[1],
    'gate_passed': sys.argv[2] == '1',
    'reason': sys.argv[3],
    'exit_code': int(sys.argv[4]),
    'checkpoint': sys.argv[5],
    'threshold_gb': float(sys.argv[6]),
    'initial_available_gb': float(sys.argv[7]),
    'final_available_gb': float(sys.argv[8]),
    'purge': {
        'attempted': sys.argv[9] == '1',
        'success': sys.argv[10] == '1',
        'recovered_mb': float(sys.argv[11]),
        'skipped_reason': sys.argv[12],
        'cold_cache_effect': 'subsequent SSD reads will be from physical flash until caches repopulate' if sys.argv[10] == '1' else 'none'
    },
    'initial_memory': json.loads(sys.argv[13]),
    'final_memory': json.loads(sys.argv[14])
}
print(json.dumps(data, indent=2))
" "${STATUS}" "${GATE_PASSED}" "${REASON}" "${EXIT_CODE}" "${CHECKPOINT}" "${THRESHOLD_GB}" \
  "${INITIAL_AVAIL_GB}" "${FINAL_AVAIL_GB}" "${PURGE_ATTEMPTED}" "${PURGE_SUCCESS}" \
  "${PURGE_RECOVERED_MB}" "${PURGE_SKIPPED_REASON}" "${MEM_INITIAL}" "${MEM_FINAL}"
else
    echo "=== Veystrix Safe Memory Preflight ==="
    echo "Status:               ${STATUS}"
    echo "Reason:               ${REASON}"
    echo "Checkpoint:           ${CHECKPOINT}"
    echo "Startup Threshold:    ${THRESHOLD_GB} GB"
    echo "Available Memory:     ${FINAL_AVAIL_GB} GB (Initial: ${INITIAL_AVAIL_GB} GB)"
    echo "Memory Pressure:      Level ${FINAL_PRESSURE}"
    echo "Swap Used:            ${INITIAL_SWAP_USED} MB"
    echo "File-backed Memory:   ${FILE_BACKED_MB} MB"
    if [[ "${PURGE_ATTEMPTED}" == "1" ]]; then
        echo "Purge Attempted:      Yes (Success: ${PURGE_SUCCESS}, Recovered: ${PURGE_RECOVERED_MB} MB)"
        if [[ "${PURGE_SUCCESS}" == "1" ]]; then
            echo "Cold-cache note:      SSD cache cleared; initial read latency may increase."
        fi
    fi
    if [[ -n "${PURGE_SKIPPED_REASON}" && "${PURGE_ATTEMPTED}" != "1" ]]; then
        echo "Purge Note:           ${PURGE_SKIPPED_REASON}"
    fi
fi

# Step 8: Optionally run model preflight check if requested and ready
if [[ "${EXIT_CODE}" -eq 0 && "${RUN_PREFLIGHT}" == "1" ]]; then
    if [[ "${JSON_OUTPUT}" != "1" ]]; then
        echo ""
        echo "--- Executing Python Preflight Check ---"
    fi
    "${PYTHON}" -m models.flashnext.tests.bench.bench_prefill_memory \
        --check-ready \
        --checkpoint "${CHECKPOINT}" \
        --start-available-gb "${THRESHOLD_GB}" \
        --min-available-gb 2.0
fi

exit "${EXIT_CODE}"
