#!/bin/sh
# usage: run_text.sh <venv python> <scripts dir> <models dir> <phase0 dir> <prompt name>...
# The reference text stage of h3-image-mlx (its int8 + ConvRot Qwen3-VL) on
# each prompt. Leaves <phase0 dir>/<prompt name>/embedding.upstream.safetensors
# and the stage's time and peak memory under <prompt name>/upstream/text/.
# H3_FLOAT_ACTIVATIONS=1 runs the same int8 weights without quantizing the
# activations (see stage_trace.py) and writes embedding.upstream-float.safetensors
# (work dir upstream/text-float).
py="$1"; scripts="$2"; models="$3"; root="$4"; shift 4
suffix=""; [ -z "$H3_FLOAT_ACTIVATIONS" ] || suffix="-float"
for name in "$@"; do
  work="$root/$name/upstream/text$suffix"
  mkdir -p "$work"
  "$py" -I "$scripts/request.py" "$root/prompts.json" "$name" "$models" 1024 1024 0 "$work/request.json"
  /usr/bin/time -l -o "$work/time.text.txt" "$py" "$scripts/stage_trace.py" text "$work" 2> "$work/stderr.text.txt" || { echo "FAILED $name text"; tail -5 "$work/stderr.text.txt"; exit 1; }
  mv "$work/embedding.safetensors" "$root/$name/embedding.upstream$suffix.safetensors"
  echo "$name text${suffix}: $(grep -E 'real' "$work/time.text.txt" | awk '{print $1}') s, peak footprint $(grep 'peak memory footprint' "$work/time.text.txt" | awk '{printf "%.2f GB", $1/1e9}')"
done
