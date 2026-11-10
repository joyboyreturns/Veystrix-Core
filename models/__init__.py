"""Model-specific runtimes. Shared chat code must not live here."""
from macqwen import env_compat as _env_compat  # noqa: F401  (VEYSTRIX_* aliases must apply before any runtime reads MACQWEN_*)
