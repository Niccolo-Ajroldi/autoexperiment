#!/usr/bin/bash

base_dir="/scratch/project_462000963/users/niccolo/pre_decay_ckpts/ckpts"

for d in "$base_dir"/*; do
  [ -d "$d" ] || continue

  # find iter_* dir
  iter_dir=$(basename "$(ls -d "$d"/iter_* 2>/dev/null | head -n1)")
  [ -n "$iter_dir" ] || continue

  # extract number (remove prefix and leading zeros)
  iter_num=${iter_dir#iter_}
  iter_num=$(echo "$iter_num" | sed 's/^0*//')

  echo "$iter_num" > "$d/latest_checkpointed_iteration.txt"
done