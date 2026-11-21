#!/bin/zsh
# Two measured generations that differ in one setting, and whether they made
# the same picture: the frames before encoding (H3SPIKE_DUMP) and the saved
# latents (H3SPIKE_SAVE_LATENT) are compared byte for byte.
#
#   tools/perf/p0_pair.sh <name> <width> <height> <frames> <steps> \
#       <VAR=value for the second run only> [VAR=value for both ...]
#
# Runs tools/perf/p0_run.sh as <name>_a (without the setting) and <name>_b
# (with it); see there for the output directory and the engine's settings.
# Prints "IDENTICAL" or "DIFFERENT" and exits 1 unless both runs finished and
# agree.
set -eu
name=$1 width=$2 height=$3 frames=$4 steps=$5 only_b=$6
shift 6
run=${0:A:h}/p0_run.sh
out=${P0_OUT:-$HOME/H3cApp-models/diffusers-p0}
mkdir -p "$out"
for side in a b; do
    extra=("$@")
    [[ $side == b ]] && extra+=("$only_b")
    rm -f "$out/${name}_$side.rgb" "$out/${name}_$side.h3lt"
    "$run" "${name}_$side" "$width" "$height" "$frames" "$steps" \
        "H3SPIKE_DUMP=$out/${name}_$side.rgb" \
        "H3SPIKE_SAVE_LATENT=$out/${name}_$side.h3lt" "${extra[@]}" \
        | grep -E "DiT arena: (total|[0-9]+ tokens)|memory: denoise enqueue|Done in|failed|exit status|^power:" \
        | sed "s/^/$side: /"
done
verdict=IDENTICAL
for kind in rgb h3lt; do
    if [[ ! -s "$out/${name}_a.$kind" || ! -s "$out/${name}_b.$kind" ]]; then
        echo "$name: no $kind output"
        verdict=DIFFERENT
    elif ! cmp -s "$out/${name}_a.$kind" "$out/${name}_b.$kind"; then
        echo "$name: $kind differs"
        verdict=DIFFERENT
    fi
done
echo "$name: $verdict ($only_b)"
[[ $verdict == IDENTICAL ]]
