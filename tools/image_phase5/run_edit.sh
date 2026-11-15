#!/bin/sh
# usage: run_edit.sh <venv python> <models dir> <work dir> <prompt> <WxH> <seed> <reference image>...
# One edit with h3-image-mlx, stage by stage (text, references, dit, vae),
# keeping every stage output, the traces of edit_stage.py and the time and
# peak memory of each stage in <work dir>.
py="$1"; models="$2"; work="$3"; prompt="$4"; size="$5"; seed="$6"; shift 6
scripts="$(cd "$(dirname "$0")" && pwd)"
w=${size%x*}; h=${size#*x}
mkdir -p "$work"
"$py" -I "$scripts/edit_request.py" "$prompt" "$models" "$w" "$h" "$seed" "$work/request.json" "$@"
for stage in text references dit vae; do
  H3_PROFILE="$work/profile.$stage.json" /usr/bin/time -l -o "$work/time.$stage.txt" "$py" "$scripts/edit_stage.py" "$stage" "$work" 2> "$work/stderr.$stage.txt" || { echo "FAILED $stage"; tail -5 "$work/stderr.$stage.txt"; exit 1; }
  echo "$stage: $(grep -E 'real' "$work/time.$stage.txt" | awk '{print $1}') s, peak footprint $(grep 'peak memory footprint' "$work/time.$stage.txt" | awk '{printf "%.2f GB", $1/1e9}')"
done
