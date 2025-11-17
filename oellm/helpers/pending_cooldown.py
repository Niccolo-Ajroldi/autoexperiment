#!/usr/bin/env python3
"""
Creates a symlink to a specific checkpoint iteration.
Checks if the stable run has reached the specified iteration.
If so, creates a symlink in the decay run pointing to that iteration.

Usage:
    pending_cooldown.py CKPT_DIR STABLE_NAME DECAY_NAME ITER

Returns:
    1 — iteration directory found; symlink created or already present
    0 — iteration directory missing (no link made)
"""


import sys
from pathlib import Path


def main():

    if len(sys.argv) < 5:
        print("Usage: pending_cooldown.py CKPT_DIR STABLE_NAME DECAY_NAME ITER", file=sys.stderr)
        print(0)
        return

    # CLI args: LOGS stable_name decay_name iter
    ckpt_dir, stable_name, decay_name, iter_str = sys.argv[1:]

    # Read and write paths
    read_folder = Path(ckpt_dir) / stable_name
    write_folder = Path(ckpt_dir) / decay_name

    # Format iteration as a 7-digit, zero-padded decimal number
    iter_dir = f"iter_{int(iter_str):07d}"

    # Target directory to look for
    target = read_folder / iter_dir

    # Check if a matching directory was found and create the symlink.
    if target.is_dir():
        write_folder.mkdir(parents=True, exist_ok=True)
        link_path = write_folder / iter_dir
        if link_path.exists() or link_path.is_symlink():
            # Symlink or file already exists — nothing to do
            print(1)
            sys.exit(0)
        link_path.symlink_to(target.resolve())
        (write_folder / "latest_checkpointed_iteration.txt").write_text(f"{iter_str}\n")
        print(1)
        sys.exit(0)

    else:
        print(0)
        sys.exit(0)


if __name__ == "__main__":
    main()
