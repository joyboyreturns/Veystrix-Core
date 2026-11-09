# Tests must never write client prompts into the owner's real ~/.flashnext/warm.
import atexit
import os
import shutil
import tempfile

if "VEYSTRIX_WARM_DIR" not in os.environ:
    _warm_dir = tempfile.mkdtemp(prefix="veystrix-test-warm-")
    os.environ["VEYSTRIX_WARM_DIR"] = _warm_dir
    atexit.register(shutil.rmtree, _warm_dir, ignore_errors=True)
