"""Unit tests. They must not depend on the machine they run on, including how much disk it has free."""
import os

# The prefix cache refuses to write to SSD unless this much disk is free (default 50 GB). CI runners have less, and
# a developer machine may too, so persistence tests would silently see "write skipped". Tests that exercise the
# reserve pass min_disk_free_bytes explicitly.
os.environ.setdefault("FLASHNEXT_DISK_RESERVE_GB", "0")
# Tests must never write client prompts into the owner's real ~/.flashnext/warm.
import atexit
import os
import shutil
import tempfile

if "VEYSTRIX_WARM_DIR" not in os.environ:
    _warm_dir = tempfile.mkdtemp(prefix="veystrix-test-warm-")
    os.environ["VEYSTRIX_WARM_DIR"] = _warm_dir
    atexit.register(shutil.rmtree, _warm_dir, ignore_errors=True)
