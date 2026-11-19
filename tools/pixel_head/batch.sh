#!/bin/zsh
# The clips the pixel-head fit is made on and judged with: 20 to fit
# (fit_NN), 4 with other prompts and seeds to judge (held_NN). 384x384,
# 56 frames, 4 steps with the Turbo LoRA; about a minute each on an M5.
# Clips already there are kept. Stops if a measurement of the other kind
# (tools/perf) is running, so as not to disturb its times.
set -eu
here=${0:A:h}
out=${PIXEL_HEAD_OUT:-$HOME/H3cApp-models/pi-h3-prep}
export PIXEL_HEAD_LORA=${PIXEL_HEAD_LORA:-$HOME/Library/Application Support/H3cApp/LoRA/minimax_h3_fl2v_turbo_4step_v1.2_768p_bf16.safetensors}
prompts=(
  "A red fox trotting through fresh snow in a birch forest, morning light."
  "A woman with curly hair laughing at a cafe table, shallow depth of field."
  "Waves breaking on black volcanic rocks at sunset, spray in the air."
  "A busy night street in the rain, neon signs reflected on wet asphalt."
  "A close-up of hands kneading bread dough on a floured wooden table."
  "A steam locomotive crossing a stone viaduct in green hills."
  "A tabby cat stretching on a sunny windowsill beside potted herbs."
  "An old man reading a newspaper on a park bench, autumn leaves falling."
  "A field of sunflowers swaying in the wind under a blue sky with clouds."
  "A chef flipping vegetables in a flaming wok in a dark kitchen."
  "A sailboat on a calm lake with snowy mountains behind, aerial view."
  "A child in a yellow raincoat jumping in puddles on a cobbled street."
  "A hummingbird hovering at a red flower, slow motion, blurred garden."
  "A library aisle with tall wooden shelves, camera moving slowly forward."
  "A drummer playing on a small stage with coloured lights and haze."
  "A herd of horses galloping across a dusty plain at golden hour."
  "A bowl of ramen with rising steam, chopsticks lifting noodles."
  "A city skyline at dusk with traffic on a bridge, time-lapse clouds."
  "A potter shaping a clay vase on a spinning wheel, close view."
  "A koi pond with ripples and floating maple leaves, seen from above."
  "A golden retriever running along a beach, splashing through shallow water."
  "A street musician playing violin in a subway station as people pass."
  "A campfire at night with sparks rising, pine trees in the dark."
  "A market stall with piles of colourful spices, a hand scooping saffron."
)
for index in {1..24}; do
    if (( index <= 20 )); then name=$(printf 'fit_%02d' $index); seed=$((100 + index))
    else name=$(printf 'held_%02d' $((index - 20))); seed=$((900 + index)); fi
    [[ -s $out/$name.bf16 && -s $out/$name.rgb ]] && continue
    if pgrep -f 'chain_p1[a-z]*\.sh|p0_run\.sh' > /dev/null; then
        echo "a measurement is running: stopped before $name"; exit 2
    fi
    echo "$name: $(date +%H:%M:%S)"
    "$here/run.sh" $name 384 384 56 4 $seed "$prompts[$index]"
    rm -f "$out/$name.mp4"
done
echo "all clips are there"
