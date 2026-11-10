"""The chat, shared by every model.

Pure Python on purpose. The 27B and Flash-Next runtimes need different
Python environments, so this package must import in both. Anything that
touches MLX belongs in a backend, not here. See docs/RESTRUCTURE.md.

Veystrix is the current name of this project (formerly MACQWEN). The import name
``macqwen`` is kept so existing scripts, caches and harness clients keep working;
``import veystrix`` is an alias for it. See docs/branding/MIGRATION.md.
"""
from macqwen import env_compat as _env_compat  # noqa: F401  (applies VEYSTRIX_* aliases on import)
