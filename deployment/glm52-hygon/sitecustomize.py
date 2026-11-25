"""Select the frozen GLM source tree without modifying system editable installs."""
import os
import sys

if os.environ.get("GLM52_INTEGRATED_RUNTIME") == "1":
    # scikit-build's editable finder precedes PYTHONPATH; remove it only in
    # this explicitly selected runtime, including spawned worker interpreters.
    sys.meta_path[:] = [
        finder for finder in sys.meta_path
        if type(finder).__module__ != "_editable_skbc_flag_gems"
    ]
