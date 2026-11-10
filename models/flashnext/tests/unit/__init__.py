"""Unit tests. They must not depend on the machine they run on, including how much disk it has free."""
import os

# The prefix cache refuses to write to SSD unless this much disk is free (default 50 GB). CI runners have less, and
# a developer machine may too, so persistence tests would silently see "write skipped". Tests that exercise the
# reserve pass min_disk_free_bytes explicitly.
os.environ.setdefault("FLASHNEXT_DISK_RESERVE_GB", "0")
