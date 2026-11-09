#!/bin/bash
# Start the Veystrix AGENTIC server yourself (supervised launch; swap guard, chunked prefill, SSD cache, watchdog).
#
#   ./serve.sh                 one supervised run in this terminal (Ctrl-C stops it)
#   ./serve.sh --overnight     keeps the server available for hours: restarts after a crash or a memory guard stop,
#                              bounded and logged; stays awake (caffeinate); never retries a request
#   ./serve.sh --status        what the supervisor did (last events, restarts, current state)
#   ./serve.sh --warm [...]    pre-cache every client's system prompt for every reasoning effort (run it while idle, e.g. before
#                              bed; --efforts low,xhigh  --client qwencode  --last medium  --dry-run). off == medium, so it is free.
#
# Options: --port N  --pool-gb 0|1|2  --checkpoint DIR  --max-restarts N  --max-guard-stops N  (see --help)
#          --enable layer-major   opt-in layer-major prefill (docs/flashnext/architecture-overhaul.md); default is chunk-major
# Environment: VEYSTRIX_CACHE_GB (SSD cache quota, default 50), VEYSTRIX_POOL_GB, VEYSTRIX_PORT, VEYSTRIX_API_KEY.
# Keep the Mac on AC power with Low Power Mode off. Logs: ~/.flashnext/logs/
cd "$(dirname "$0")" || exit 1
export VEYSTRIX_API_KEY="${VEYSTRIX_API_KEY:-local}"
if [[ "$1" == "--warm" ]]; then
  shift
  exec /usr/bin/python3 -m macqwen.warm "$@"
fi
if [[ "$*" == *"--status"* || "$*" == *"--help"* || "$*" == *"-h"* ]]; then
  exec /usr/bin/python3 -m models.flashnext.overnight "$@"
fi
# -i: no idle sleep, -m: no disk idle sleep, -s: no system sleep on AC power.
exec /usr/bin/caffeinate -ims /usr/bin/python3 -m models.flashnext.overnight "$@"
