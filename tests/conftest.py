"""Pytest configuration shared by the whole suite.

spconv/cumm on some environments (notably very new Colab runtimes) try to
JIT-compile `cumm` at import and fail on missing headers. Forcing the prebuilt
CUDA kernels via these env vars — set BEFORE spconv is ever imported — skips the
broken JIT path. Harmless everywhere else (the prebuilt `.so` is simply used).
"""

import os

os.environ.setdefault("CUMM_DISABLE_JIT", "1")
os.environ.setdefault("SPCONV_DISABLE_JIT", "1")
