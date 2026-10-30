"""Converts the pinned Haas-ja checkpoint to the one GGUF file the C++ port reads, with reference/fastconformer/convert.py.

usage: uv run python convert.py <model> <out dir>

Haas-ja is a FastConformer with a TDT decoder whose prediction network is Kimi Delta Attention, which the
FastConformer converter writes; it runs here, in this folder's environment and with this folder's pins, since the
checkpoint needs NeMo 2.7 and the haas_asr package, which the FastConformer environment does not hold.
"""

import os
import runpy
import sys

here = os.path.dirname(os.path.abspath(__file__))
# The converter imports pins from the first folder on the path, which is this one: run_path() adds a script's own folder
# to none.
sys.path.insert(0, here)
runpy.run_path(os.path.join(here, "..", "fastconformer", "convert.py"), run_name="__main__")
