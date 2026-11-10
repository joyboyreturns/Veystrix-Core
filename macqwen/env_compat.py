"""Accept ``VEYSTRIX_*`` names for the legacy ``MACQWEN_*`` environment variables.

The project is now called Veystrix. The engine still reads the legacy names in
many places, so existing scripts, harness clients and shells keep working. This
module copies a ``VEYSTRIX_<NAME>`` value onto ``MACQWEN_<NAME>`` for every legacy
name the engine reads. If both are set, the Veystrix name wins.

It touches nothing else: ``VEYSTRIX_*`` variables that are not legacy aliases
(``VEYSTRIX_API_KEY``, ``VEYSTRIX_MODE``, ...) are left alone, and caches,
sessions and checkpoints are not affected.
"""
from __future__ import annotations

import os
from collections.abc import MutableMapping

# Every MACQWEN_ name read by the source. The drift test in
# macqwen/tests/test_veystrix_migration.py fails if the source reads one that is missing.
LEGACY_SUFFIXES = frozenset({
    "ALLOW_REAP_XHIGH", "BONSAI2_KV", "BONSAI2_MODEL", "BONSAI2_PARENT_TIMEOUT_SECONDS", "BONSAI2_PYTHON",
    "BONSAI2_QMM_DEADLINE_S", "BONSAI2_QMM_MAX_ADDITIONAL_MB", "FADE_MS", "FLASHNEXT_MODEL", "FLASHNEXT_PYTHON",
    "K2_HORIZON_MODEL", "K2_HORIZON_PYTHON", "MODEL", "MODEL_LOCK", "MODEL_ROOT", "PROFILE_EPHEMERAL", "PYTHON",
    "QWEN27B_MODEL", "QWEN27B_PYTHON", "REAP_XHIGH_THINK_BUDGET", "REQUEST_LOG", "RESULTS_DIR", "SERVER_API_KEY",
    "TOKENIZER_MODEL", "WORKSPACE",
})
# Names the source builds from a prefix (for example ``f"MACQWEN_BONSAI2_{name}"``).
LEGACY_PREFIXES = ("BONSAI2_", "K2_HORIZON_", "QWEN27B_")

_NEW, _OLD = "VEYSTRIX_", "MACQWEN_"


def apply(environ: MutableMapping[str, str] | None = None) -> list[str]:
    """Copy VEYSTRIX_<NAME> onto MACQWEN_<NAME>. Returns the legacy names that were set."""
    env = os.environ if environ is None else environ
    changed = []
    for name in list(env):
        if not name.startswith(_NEW):
            continue
        suffix = name[len(_NEW):]
        if suffix in LEGACY_SUFFIXES or suffix.startswith(LEGACY_PREFIXES):
            legacy = _OLD + suffix
            if env.get(legacy) != env[name]:
                env[legacy] = env[name]
                changed.append(legacy)
    return changed


apply()
