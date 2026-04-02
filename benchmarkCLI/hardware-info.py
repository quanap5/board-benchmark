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
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
        )
        for line in result.stdout.strip().split("\n")[1:]:
            parts = line.split()
            if len(parts) >= 6:
                disks.append({
                    "device": parts[0], "size": parts[1], "used": parts[2],
                    "available": parts[3], "use_pct": parts[4], "mount": parts[5],
                })
    except FileNotFoundError:
        pass
    return disks


def get_gpu_info():
    """Get GPU info via nvidia-smi, Tegra SoC detection, or lspci."""
    gpus = []

    # 1. Desktop NVIDIA GPU (nvidia-smi)
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3:
                    gpus.append({"name": parts[0], "memory": parts[1], "driver": parts[2]})
            return gpus
    except FileNotFoundError:
        pass

    # 2. Jetson Tegra GPU (integrated SoC, not on PCIe bus)
    if os.path.exists("/etc/nv_tegra_release"):
        gpu = _get_tegra_gpu()
        if gpu:
            return [gpu]

    # 3. Fallback: lspci for discrete GPUs
    try:
        result = subprocess.run(
            ["lspci"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True
        )
        for line in result.stdout.split("\n"):
            if "VGA" in line or "3D controller" in line:
                gpus.append({"name": line.split(": ", 1)[-1]})
    except FileNotFoundError:
        pass
    return gpus


def _get_tegra_gpu():
    """Detect Tegra integrated GPU from sysfs and /etc/nv_tegra_release."""
    try:
        with open("/etc/nv_tegra_release") as f:
            release = f.read().strip()
    except (IOError, OSError):
        release = ""

    names = {"33": "Tegra X1 (Maxwell, 256 CUDA)", "24": "Tegra X2 (Pascal, 256 CUDA)",
             "25": "Tegra Xavier (Volta, 512 CUDA)", "35": "Tegra Orin (Ampere)"}

    chip_id = ""
    for p in ["/sys/module/tegra_fuse/parameters/tegra_chip_id", "/sys/devices/soc0/soc_id"]:
        try:
            with open(p) as f:
                chip_id = f.read().strip()
                break
        except (IOError, OSError):
            continue

    name = names.get(chip_id, "Tegra X2 (Pascal, 256 CUDA)" if "R32" in release else "Tegra GPU")
    gpu = {"name": "NVIDIA " + name, "type": "Integrated (SoC)"}

    mem = get_memory_info()
    if mem.get("total_gb"):
        gpu["memory"] = "{} MB (shared)".format(int(mem["total_gb"] * 1024))
    if release:
        gpu["driver"] = "L4T " + release.split(",")[0].strip()
    return gpu


def print_section(title, items):
    """Print a formatted section."""
    print(f"\n{'=' * 50}\n  {title}\n{'=' * 50}")
    for key, value in items:
        print(f"  {key:<20} {value}")


def main():
    print("\n  HARDWARE SYSTEM INFORMATION")

    print_section("OS / Kernel", [
        ("Hostname:", platform.node()), ("System:", platform.system()),
        ("Kernel:", platform.release()), ("Version:", platform.version()),
        ("Architecture:", platform.machine()),
    ])

    cpu = get_cpu_info()
    print_section("CPU", [
        ("Model:", cpu["name"]), ("Logical CPUs:", cpu["logical_cpus"]),
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
        print_section("Disks", [
            (f"{d['mount']}:", f"{d['used']}/{d['size']} ({d['use_pct']}) - {d['device']}")
            for d in disks
        ])

    gpus = get_gpu_info()
    if gpus:
        gpu_items = []
        for i, g in enumerate(gpus):
            gpu_items.append((f"GPU {i}:", g["name"]))
            for key in ["type", "memory", "driver"]:
                if key in g:
                    gpu_items.append((f"  {key.title()}:", g[key]))
        print_section("GPU", gpu_items)
    else:
        print_section("GPU", [("Status:", "No GPU detected")])

    print()


if __name__ == "__main__":
    main()
