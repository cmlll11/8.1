from __future__ import annotations

import argparse
import subprocess


def main():
    parser = argparse.ArgumentParser(description="Select the least-used CUDA device")
    parser.add_argument("--max-used-mib", type=int, default=2048)
    args = parser.parse_args()
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            text=True,
        )
        rows = []
        for line in output.splitlines():
            index, used = [part.strip() for part in line.split(",", 1)]
            rows.append((int(used), int(index)))
        if not rows:
            raise RuntimeError("nvidia-smi returned no GPUs")
        below = [row for row in rows if row[0] <= int(args.max_used_mib)]
        used, index = min(below or rows)
        print(f"cuda:{index}")
    except Exception:
        print("cpu")


if __name__ == "__main__":
    main()

