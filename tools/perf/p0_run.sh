#!/bin/zsh
# One measured generation for plans' Diffusers-adoption baseline (P0), with the
# engine set up the way the app's "Fast, int8 cache" mode sets it up
# (H3Engine.swift: the int8 cache, MLP weights streamed from it, text-encoder
# prefetch depth 1) and with every approximation off (denoise reuse 1, core
# reuse 1, no token reduction - the CLI's defaults).
#
#   tools/perf/p0_run.sh <name> <width> <height> <frames> <steps> [VAR=value ...]
#
# Writes <name>.log, <name>.mp4 and <name>.json to $P0_OUT (default
# ~/H3cApp-models/diffusers-p0, outside the synced ~/Documents). Extra
# VAR=value arguments are exported for the run (H3_SDPA_MODE=chunked, ...).
# H3_MEMORY_LIMIT_GIB ends the process before the Mac runs out of memory.
# The power source is recorded before and after: compare times only between
# runs on AC power.
set -eu
name=$1 width=$2 height=$3 frames=$4 steps=$5
shift 5
root=${0:A:h:h:h}
out=${P0_OUT:-$HOME/H3cApp-models/diffusers-p0}
model=${P0_MODEL:-$HOME/Library/Application Support/H3cApp/Imported/MiniMax-H3 GGUF}
cache=${P0_CACHE:-$HOME/Library/Application Support/H3cApp/cache/B1596340-9D23-4162-8168-FA732395DFC3/dit_int8_v2.cache}
prompt=${P0_PROMPT:-A cat playing with a ball of yarn.}
mkdir -p "$out"
# Time only compares between runs on the same power source: this Mac turns
# low power mode on while on battery, which makes the DiT about 1.7x slower.
power() { echo "$(pmset -g batt | head -1 | tr -d "'"), low power mode $(pmset -g | awk '/lowpowermode/{print $2}')"; }
power_before=$(power)

export H3_ATTENTION_CACHE=$cache H3_INT8_STREAM_MLP=1 H3_QWEN_PREFETCH_DEPTH=1
export H3_MEMORY_TRACE=1 H3_PROFILE=1 H3_DIT_ARENA_TRACE=1 H3_SDPA_LOG=1
export H3_MEMORY_LIMIT_GIB=${H3_MEMORY_LIMIT_GIB:-22}
export H3SPIKE_WIDTH=$width H3SPIKE_HEIGHT=$height H3SPIKE_FRAMES=$frames
export H3SPIKE_STEPS=$steps H3SPIKE_SEED=${P0_SEED:-7}
extra=()
for assignment in "$@"; do
    export "$assignment"
    extra+=("\"$assignment\"")
done

cat > "$out/$name.json" <<EOF
{
  "name": "$name",
  "commit": "$(git -C "$root" rev-parse HEAD)",
  "dirty": $(git -C "$root" diff --quiet && echo false || echo true),
  "os": "$(sw_vers -productVersion) ($(sw_vers -buildVersion))",
  "chip": "$(sysctl -n machdep.cpu.brand_string)",
  "memory_bytes": $(sysctl -n hw.memsize),
  "model": "$model",
  "attention_cache": "$cache",
  "prompt": "$prompt",
  "width": $width, "height": $height, "frames": $frames, "steps": $steps,
  "seed": $H3SPIKE_SEED,
  "denoise_reuse": ${H3SPIKE_REUSE:-1}, "core_reuse": ${H3SPIKE_CORE_REUSE:-1},
  "token_reduction": ${H3SPIKE_TOKEN_REDUCTION:-0},
  "fast_attention": ${H3SPIKE_FAST_ATTENTION:-0},
  "extra_environment": [${(j:, :)extra}],
  "power_before": "$power_before",
  "swap_before": "$(sysctl -n vm.swapusage)"
}
EOF
"$root/build/bin/h3_generate_cli" "$model" "$out/$name.mp4" "$prompt" \
    > "$out/$name.log" 2>&1 || echo "exit status $?" >> "$out/$name.log"
echo "swap after: $(sysctl -n vm.swapusage)" >> "$out/$name.log"
echo "power: before $power_before; after $(power)" >> "$out/$name.log"
grep -E "plan:|memory:|profile:|DiT arena|SDPA|sdpa|Done in|failed|exit status|swap after|^power:" \
    "$out/$name.log" | cut -c1-240
