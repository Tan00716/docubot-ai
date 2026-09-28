"""Resource measurements for the model comparison: memory and disk (no extra packages).

Memory comes from the operating system, not from Python: the ONNX model's
weights live in native memory that Python's own counters cannot see.

    working set    - RAM the process occupies right now
    private bytes  - memory the process has committed (RAM or page file)
    peak ...       - the highest value since the process started

On Windows these are read with the documented Win32 calls
GetProcessMemoryInfo / GlobalMemoryStatusEx. Elsewhere only the peak
resident size is available (resource.getrusage); the rest is None, meaning
"not measured", never 0. Every model is measured in its own fresh process,
so one model's peak cannot hide inside another's.
"""

import ctypes
import os
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProcessMemory:
    working_set: int | None
    peak_working_set: int | None
    private_bytes: int | None
    peak_private_bytes: int | None


@dataclass(frozen=True)
class SystemMemory:
    total_physical: int | None
    available_physical: int | None
    available_commit: int | None  # how much more memory may be committed (RAM + page file)


class _ProcessMemoryCounters(ctypes.Structure):
    """Win32 PROCESS_MEMORY_COUNTERS_EX (field order as documented by Microsoft)."""

    _fields_ = [
        ("cb", ctypes.c_uint32),
        ("PageFaultCount", ctypes.c_uint32),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


class _MemoryStatus(ctypes.Structure):
    """Win32 MEMORYSTATUSEX."""

    _fields_ = [
        ("dwLength", ctypes.c_uint32),
        ("dwMemoryLoad", ctypes.c_uint32),
        ("ullTotalPhys", ctypes.c_uint64),
        ("ullAvailPhys", ctypes.c_uint64),
        ("ullTotalPageFile", ctypes.c_uint64),
        ("ullAvailPageFile", ctypes.c_uint64),
        ("ullTotalVirtual", ctypes.c_uint64),
        ("ullAvailVirtual", ctypes.c_uint64),
        ("ullAvailExtendedVirtual", ctypes.c_uint64),
    ]


def process_memory() -> ProcessMemory:
    if sys.platform == "win32":
        counters = _ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi = ctypes.windll.psapi
        psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32]
        if psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters),
                                      counters.cb):
            return ProcessMemory(counters.WorkingSetSize, counters.PeakWorkingSetSize,
                                 counters.PrivateUsage, counters.PeakPagefileUsage)
        return ProcessMemory(None, None, None, None)
    try:
        import resource  # POSIX only

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak *= 1 if sys.platform == "darwin" else 1024  # macOS reports bytes, Linux KiB
        return ProcessMemory(None, peak, None, None)
    except (ImportError, OSError):
        return ProcessMemory(None, None, None, None)


def system_memory() -> SystemMemory:
    if sys.platform != "win32":
        return SystemMemory(None, None, None)
    status = _MemoryStatus()
    status.dwLength = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return SystemMemory(None, None, None)
    return SystemMemory(status.ullTotalPhys, status.ullAvailPhys, status.ullAvailPageFile)


def files_size(folder: Path, names: Iterable[str]) -> int:
    """Total size of the named files (what a cold download of exactly them fetches)."""
    return sum((folder / name).stat().st_size for name in names)


def folder_size(folder: Path) -> int:
    """Disk use of a folder. Symbolic links count as links, so a file is never counted twice."""
    total = 0
    for root, _, names in os.walk(folder):
        for name in names:
            total += (Path(root) / name).stat(follow_symlinks=False).st_size
    return total
