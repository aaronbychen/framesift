"""Opt this process out of Windows 11 power throttling (EcoQoS).

Windows throttles processes that own no foreground window: their threads can land on efficiency
cores at low clocks. Token-by-token generation is bound by the CPU thread that launches GPU
kernels, so a throttled process starves the GPU, which then drops to idle clocks as well
(measured here: ~300 ms/token throttled vs ~75 ms/token unthrottled on the same frames).
This only changes this process's own scheduling hints; no system setting is touched.
"""
import ctypes
import os
import sys

PROCESS_POWER_THROTTLING = 4  # PROCESS_INFORMATION_CLASS.ProcessPowerThrottling
THROTTLE_EXECUTION_SPEED = 0x1
THROTTLE_IGNORE_TIMER_RESOLUTION = 0x4
ABOVE_NORMAL_PRIORITY_CLASS = 0x00008000


def disable_power_throttling():
    """Returns a dict describing what was applied (empty off Windows)."""
    if sys.platform != "win32":
        return {}
    from ctypes import wintypes

    class State(ctypes.Structure):
        _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    k32.SetProcessInformation.restype = wintypes.BOOL
    k32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    k32.SetPriorityClass.restype = wintypes.BOOL
    k32.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
    k32.SetProcessAffinityMask.restype = wintypes.BOOL

    proc = k32.GetCurrentProcess()
    state = State(1, THROTTLE_EXECUTION_SPEED | THROTTLE_IGNORE_TIMER_RESOLUTION, 0)
    applied = {
        "ecoqos_off": bool(k32.SetProcessInformation(proc, PROCESS_POWER_THROTTLING, ctypes.byref(state),
                                                     ctypes.sizeof(state))),
        "above_normal_priority": bool(k32.SetPriorityClass(proc, ABOVE_NORMAL_PRIORITY_CLASS)),
    }
    # Optional: pin to the first N logical CPUs (P-cores come first on Intel hybrid parts).
    pcores = int(os.getenv("PIN_LOGICAL_CPUS", "0"))
    if pcores > 0:
        applied["affinity_cpus"] = pcores if k32.SetProcessAffinityMask(proc, (1 << pcores) - 1) else None
    return applied
