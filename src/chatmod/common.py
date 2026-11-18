"""Shared helpers: data loading and decision-threshold tuning (same procedure for every model)."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
SEED = 42
# Deployment constraint of the use case: a CPU-only game server that handles many chats at once, so each
# message gets ONE CPU core. Evaluation and the app both run PyTorch with this many threads.
# (scripts/benchmark_threads.py additionally reports latency with 1, 2 and 4 threads.)
CPU_THREADS = 1


def request_full_cpu_speed() -> bool:
    """Ask Windows not to power-throttle this process (EcoQoS).

    Windows 11 moves background processes to slower efficiency cores; the app server is in the background
    while the user types in the browser, which made its latency jump. A real moderation server would run at
    full speed. No-op (returns False) on other operating systems.
    """
    import sys

    if sys.platform != "win32":
        return False
    import ctypes
    from ctypes import wintypes

    class PowerThrottlingState(ctypes.Structure):
        _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    k32.SetProcessInformation.restype = wintypes.BOOL
    # ProcessPowerThrottling = 4; control EXECUTION_SPEED (0x1) and set it to "not throttled" (state 0).
    state = PowerThrottlingState(1, 0x1, 0x0)
    return bool(k32.SetProcessInformation(k32.GetCurrentProcess(), 4, ctypes.byref(state), ctypes.sizeof(state)))


def load_split(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / f"{name}.csv", keep_default_na=False)


def tune_threshold(y_true, p_offensive) -> tuple[float, float]:
    """Pick the decision threshold that maximizes macro-F1 on the validation split."""
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.05, 0.96, 0.01):
        f1 = f1_score(y_true, (np.asarray(p_offensive) >= t).astype(int), average="macro")
        if f1 > best_f1:
            best_t, best_f1 = round(float(t), 2), float(f1)
    return best_t, best_f1
