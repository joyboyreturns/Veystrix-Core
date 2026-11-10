"""Veystrix: local inference engine for Apple Silicon (formerly MACQWEN).

This package is the current public import name. The implementation still lives in
the ``macqwen`` package (and ``models``) so that existing imports, scripts, saved
sessions and harness clients keep working. ``veystrix.<module>`` resolves to the very
same module object as ``macqwen.<module>``.

    import veystrix.preferences      # same object as macqwen.preferences
    python -m veystrix --help        # same as ./chat.sh --help

Migration plan: docs/branding/MIGRATION.md.
"""
from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import sys

__version__ = "3.8.1.dev0"   # keep equal to pyproject.toml (checked by a test)

_PREFIX, _TARGET = "veystrix.", "macqwen."


class _AliasFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Resolve ``veystrix.X`` to the existing ``macqwen.X`` module object."""

    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith(_PREFIX):
            return None
        real = _TARGET + fullname[len(_PREFIX):]
        try:
            real_spec = importlib.util.find_spec(real)
        except (ImportError, ValueError):
            return None
        if real_spec is None:
            return None
        return importlib.machinery.ModuleSpec(
            fullname, self, is_package=real_spec.submodule_search_locations is not None)

    def create_module(self, spec):
        return importlib.import_module(_TARGET + spec.name[len(_PREFIX):])

    def exec_module(self, module):          # the real module is already executed
        return None


if not any(isinstance(finder, _AliasFinder) for finder in sys.meta_path):
    sys.meta_path.insert(0, _AliasFinder())
