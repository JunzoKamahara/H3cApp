#!/bin/sh
# usage: run_sizes.sh <venv python> <scripts dir> <models dir> <phase0 dir> <prompt name> <h3c|upstream> <seed> <WxH>...
# For each size: DiT stage (traced) and VAE stage of h3-image-mlx, driven by
# <phase0 dir>/<prompt name>/embedding.<source>.safetensors - the engine's own
# text embedding (h3_image_text_embed) or the reference one (run_text.sh).
# Keeps every stage output and the time/memory of each stage under
# <phase0 dir>/<prompt name>/<source>/<WxH>/.
py="$1"; scripts="$2"; models="$3"; root="$4"; name="$5"; source="$6"; seed="$7"; shift 7
for size in "$@"; do
  w=${size%x*}; h=${size#*x}; work="$root/$name/$source/$size"
  mkdir -p "$work"
  cp "$root/$name/embedding.$source.safetensors" "$work/embedding.safetensors"
  "$py" -I "$scripts/request.py" "$root/prompts.json" "$name" "$models" "$w" "$h" "$seed" "$work/request.json"
  for stage in dit vae; do
    H3_PROFILE="$work/profile.$stage.json" /usr/bin/time -l -o "$work/time.$stage.txt" "$py" "$scripts/stage_trace.py" "$stage" "$work" 2> "$work/stderr.$stage.txt" || { echo "FAILED $name $size $stage"; tail -5 "$work/stderr.$stage.txt"; exit 1; }
    echo "$name $source $size $stage: $(grep -E 'real' "$work/time.$stage.txt" | awk '{print $1}') s, peak footprint $(grep 'peak memory footprint' "$work/time.$stage.txt" | awk '{printf "%.2f GB", $1/1e9}')"
  done
done
