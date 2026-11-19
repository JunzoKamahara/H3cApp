#!/bin/zsh
# One generation that also keeps what a pixel head (DWD) would be fitted on:
# the last step's video rows as the video head reads them (H3_DUMP_FINAL_VIDEO)
# and the frames the video VAE decodes from that step's latent.
#
#   tools/pixel_head/run.sh <name> <width> <height> <frames> <steps> <seed> <prompt> [VAR=value ...]
#
# Writes <name>.bf16 and <name>.json (the rows and their shape), <name>.rgb
# (packed RGB24 frames), <name>.mp4 and <name>.log to $PIXEL_HEAD_OUT (default
# ~/H3cApp-models/pi-h3-prep, outside the synced ~/Documents). The engine is
# set up as the app's "Fast, int8 cache" mode sets it up. PIXEL_HEAD_LORA
# names a LoRA to apply (the Turbo LoRA for 4 steps), PIXEL_HEAD_NO_ROWS=1
# leaves the rows out (to see that the switch does not change the frames).
set -eu
name=$1 width=$2 height=$3 frames=$4 steps=$5 seed=$6 prompt=$7
shift 7
root=${0:A:h:h:h}
out=${PIXEL_HEAD_OUT:-$HOME/H3cApp-models/pi-h3-prep}
model=${PIXEL_HEAD_MODEL:-$HOME/Library/Application Support/H3cApp/Imported/MiniMax-H3 GGUF}
cache=${PIXEL_HEAD_CACHE:-$HOME/Library/Application Support/H3cApp/cache/B1596340-9D23-4162-8168-FA732395DFC3/dit_int8_v2.cache}
mkdir -p "$out"
export H3_ATTENTION_CACHE=$cache H3_INT8_STREAM_MLP=1 H3_QWEN_PREFETCH_DEPTH=1
export H3_MEMORY_LIMIT_GIB=${H3_MEMORY_LIMIT_GIB:-22}
export H3SPIKE_WIDTH=$width H3SPIKE_HEIGHT=$height H3SPIKE_FRAMES=$frames
export H3SPIKE_STEPS=$steps H3SPIKE_SEED=$seed H3SPIKE_DUMP=$out/$name.rgb
[[ -n ${PIXEL_HEAD_LORA:-} ]] && export H3SPIKE_LORA=$PIXEL_HEAD_LORA
[[ ${PIXEL_HEAD_NO_ROWS:-0} == 1 ]] || export H3_DUMP_FINAL_VIDEO=$out/$name
for assignment in "$@"; do export "$assignment"; done
"$root/build/bin/h3_generate_cli" "$model" "$out/$name.mp4" "$prompt" \
    > "$out/$name.log" 2>&1 || echo "exit status $?" >> "$out/$name.log"
grep -E "Done in|failed|error|exit status|Result:" "$out/$name.log" | cut -c1-200
