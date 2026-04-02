#!/usr/bin/env python3
"""Print hardware system information."""

import platform
import os
import subprocess

PROC_PATH = "/host/proc" if os.path.exists("/host/proc/cpuinfo") else "/proc"


def get_cpu_info():
    """Get CPU information."""
    info = {"name": platform.processor() or "Unknown"}

    try:
        with open(f"{PROC_PATH}/cpuinfo") as f:
            for line in f:
                if "model name" in line:
                    info["name"] = line.split(":")[1].strip()
                elif "cpu cores" in line:
                    info["cores_per_socket"] = int(line.split(":")[1].strip())
                elif "siblings" in line:
                    info["threads_per_socket"] = int(line.split(":")[1].strip())
    except FileNotFoundError:
        pass

    info["logical_cpus"] = os.cpu_count() or 0
    return info


def get_memory_info():
    """Get memory information in GB."""
    info = {}
    try:
        with open(f"{PROC_PATH}/meminfo") as f:
            for line in f:
                key, value = line.split(":")
                kb = int(value.strip().split()[0])
                if key == "MemTotal":
                    info["total_gb"] = round(kb / 1024 / 1024, 2)
                elif key == "MemAvailable":
                    info["available_gb"] = round(kb / 1024 / 1024, 2)
                elif key == "SwapTotal":
                    info["swap_total_gb"] = round(kb / 1024 / 1024, 2)
    except FileNotFoundError:
        pass
    return info


def get_disk_info():
    """Get disk usage for mounted filesystems."""
    disks = []
    try:
        result = subprocess.run(
            ["df", "-h", "--type=ext4", "--type=vfat", "--type=ntfs",
             "--type=fuseblk", "--type=xfs", "--type=btrfs"],
            capture_output=True, text=True
        )
        for line in result.stdout.strip().split("\n")[1:]:
            parts = line.split()
            if len(parts) >= 6:
                disks.append({
                    "device": parts[0],
                    "size": parts[1],
                    "used": parts[2],
                    "available": parts[3],
                    "use_pct": parts[4],
                    "mount": parts[5],
                })
    except FileNotFoundError:
        pass
    return disks


def get_gpu_info():
    """Get GPU information via lspci or nvidia-smi."""
    gpus = []
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3:
                    gpus.append({
                        "name": parts[0],
                        "memory": parts[1],
                        "driver": parts[2],
                    })
            return gpus
    except FileNotFoundError:
        pass

    try:
        result = subprocess.run(
            ["lspci"], capture_output=True, text=True
        )
        for line in result.stdout.split("\n"):
            if "VGA" in line or "3D controller" in line:
                gpus.append({"name": line.split(": ", 1)[-1]})
    except FileNotFoundError:
        pass
    return gpus


def get_os_info():
    """Get OS and kernel information."""
    return {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "architecture": platform.machine(),
        "hostname": platform.node(),
    }


def print_section(title, items):
    """Print a formatted section."""
    print(f"\n{'=' * 50}")
    print(f"  {title}")
    print(f"{'=' * 50}")
    for key, value in items:
        print(f"  {key:<20} {value}")


def main():
    print("\n  HARDWARE SYSTEM INFORMATION")

    os_info = get_os_info()
    print_section("OS / Kernel", [
        ("Hostname:", os_info["hostname"]),
        ("System:", os_info["system"]),
        ("Kernel:", os_info["release"]),
        ("Version:", os_info["version"]),
        ("Architecture:", os_info["architecture"]),
    ])

    cpu = get_cpu_info()
    print_section("CPU", [
        ("Model:", cpu["name"]),
        ("Logical CPUs:", cpu["logical_cpus"]),
        ("Cores/Socket:", cpu.get("cores_per_socket", "N/A")),
        ("Threads/Socket:", cpu.get("threads_per_socket", "N/A")),
    ])

    mem = get_memory_info()
    if mem:
        used = round(mem.get("total_gb", 0) - mem.get("available_gb", 0), 2)
        print_section("Memory", [
            ("Total:", f"{mem.get('total_gb', 'N/A')} GB"),
            ("Available:", f"{mem.get('available_gb', 'N/A')} GB"),
            ("Used:", f"{used} GB"),
            ("Swap:", f"{mem.get('swap_total_gb', 'N/A')} GB"),
        ])

    disks = get_disk_info()
    if disks:
        disk_items = []
        for d in disks:
            disk_items.append((
                f"{d['mount']}:",
                f"{d['used']}/{d['size']} ({d['use_pct']}) - {d['device']}"
            ))
        print_section("Disks", disk_items)

    gpus = get_gpu_info()
    if gpus:
        gpu_items = []
        for i, g in enumerate(gpus):
            gpu_items.append((f"GPU {i}:", g["name"]))
            if "memory" in g:
                gpu_items.append(("  Memory:", g["memory"]))
            if "driver" in g:
                gpu_items.append(("  Driver:", g["driver"]))
        print_section("GPU", gpu_items)
    else:
        print_section("GPU", [("Status:", "No GPU detected")])

    print()


if __name__ == "__main__":
    main()
