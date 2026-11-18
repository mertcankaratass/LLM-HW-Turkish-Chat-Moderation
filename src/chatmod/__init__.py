import os
import sys
import warnings

# We only use PyTorch. Stop transformers from also loading TensorFlow when it happens to be installed.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")

# Harmless deprecation notice printed by torch.load for the INT8 state dict (PyTorch internals, not our code).
warnings.filterwarnings("ignore", message="TypedStorage is deprecated")

# Result tables contain characters such as "≤" and "−": print them as UTF-8 even when Windows redirects the
# output to a file or pipe with a legacy code page (e.g. cp1254), instead of crashing at the final print.
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure") and (_stream.encoding or "").lower() != "utf-8":
        _stream.reconfigure(encoding="utf-8", errors="replace")
