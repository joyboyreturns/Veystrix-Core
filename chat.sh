#!/bin/bash
# One chat interface. macqwen/cli.py prepares and reuses the managed runtime.
# Two modes share one runtime: ./chat.sh --mode chat (default) and ./chat.sh --mode agentic (local server).
# See docs/flashnext/modes.md.
veystrix_safe_chat="${FLASHNEXT_SAFE_CHAT:-0}"
if [[ -f "$HOME/.frankenstein/env" ]]; then
  set -a; source "$HOME/.frankenstein/env"; set +a
fi
if [[ "$veystrix_safe_chat" == "1" ]]; then
  # Ignore inherited research flags, including those in the optional env file.
  # Normal backend defaults are still applied by macqwen/cli.py.
  for veystrix_key in ${!FLASHNEXT_@}; do unset "$veystrix_key"; done
  export FLASHNEXT_CHAT_GUARD=1 FLASHNEXT_EXPERT_POOL_GB=2
  export FLASHNEXT_PREFIX_CACHE=0 FLASHNEXT_PREFIX_PROFILES=0
  export FLASHNEXT_MEMORY_MANAGER=0 FLASHNEXT_AUTO_MEMORY_RECLAIM=0
  export FLASHNEXT_PREFILL_RELEASE_DECODE_POOL=0 FLASHNEXT_PREFILL_RING_EXPERTS=0
  export FLASHNEXT_PREFILL_CHUNK_TOKENS=0 FLASHNEXT_NATIVE_MTP=off
fi
exec /usr/bin/python3 "$(dirname "$0")/macqwen/cli.py" "$@"
