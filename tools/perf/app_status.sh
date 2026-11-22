#!/bin/zsh
# Runs H3cApp through a list of video generations over its local API and
# records, for each, whether it worked, how long it took and how much memory
# it used. It needs nothing installed: only macOS (15 or later, as the app
# does) and the app with a model in it. No Python, no Command Line Tools.
#
#     zsh app_status.sh /Applications/H3cApp.app <work dir> [--quick] [--only=a,b] [--redo] [--limit=minutes] [--list]
#                        [--port=number] [--lora=file]...
#
# or, for a development build (from the repository, which then gives the
# shaders and the other resources):
#
#     make libh3.a CCV_DIR=...
#     swift build --package-path native/H3Spike --scratch-path "${TMPDIR}H3cApp-build-scratch" -c release --product H3cApp
#     zsh tools/perf/app_status.sh "${TMPDIR}H3cApp-build-scratch/release/H3cApp" <work dir> ...
#
#     python3 tools/perf/app_status_report.py <work dir> <report.md>
#
# The app under test is a copy of the executable named H3cAppDevCheck, run
# with CFFIXED_USER_HOME set to <work dir>/home (as tools/image_phase1 does):
# its Application Support, its projects and its preferences (the domain
# H3cAppDevCheck, removed before and after) are its own. A copy out of an app
# bundle is signed again, ad hoc: the bundle's signature holds only inside the
# bundle, and a bundle run in place would use the installed app's settings.
# From the installed app it takes, and only reads: the model list, the LoRA
# list, the int8 caches (as symbolic links) and the denoise preview's decoder
# (a copy). Port 8420 must be free, so quit H3cApp first - or, with a build
# that reads H3_API_PORT (after v0.11.0), give the copy a port of its own with
# --port and leave H3cApp open. --lora adds a LoRA file to the copy's list
# only (the pi-H3 jobs want the pi-H3 adapters, which the installed app may
# not have).
#
# The model is the one selected in the installed app (another when that is a
# HyperFlow H3 Flash, which has a job of its own). A job whose model or LoRA
# is not there is recorded as skipped, not failed, and so is one that asks for
# the int8 cache where there is none; a job that does not name a compute mode
# runs with SSD streaming there. --quick is seven short jobs, 25 to 30 minutes
# on an M5, for a slow Mac or a first look; --limit is how long one job may
# take (60 minutes).
#
# Each job's record is a line of <work dir>/results.jsonl, with the Mac and
# the app's version in it; a job already there is skipped unless --redo. The
# app is started once and kept, as a user would keep it; if it exits, the job
# is recorded as failed and the app is started again. Keep the work dir
# outside ~/Documents.
#
# Only what every Mac has is used, and only from the system's own folders:
# curl and jq for the API and the records, od for the MP4 and WAV files,
# defaults and plutil for the preferences, qlmanage, sips and afconvert for
# the input pictures and sound, footprint for the memory, codesign.
emulate -R zsh
unalias -m '*' 2>/dev/null
PATH=/usr/bin:/bin:/usr/sbin:/sbin
zmodload zsh/datetime

arguments=() only=() extra_loras=() limit=60 redo=0 quick=0 list=0 port=8420
for argument in "$@"; do
    case $argument in
        --only=*) only=(${(s:,:)${argument#--only=}}) ;;
        --limit=*) limit=${argument#--limit=} ;;
        --redo) redo=1 ;;
        --quick) quick=1 ;;
        --list) list=1 ;;
        --port=*) port=${argument#--port=} ;;
        --lora=*) extra_loras+=("${${argument#--lora=}:A}") ;;
        --*) ;;
        *) arguments+=("$argument") ;;
    esac
done
typeset -F limit_seconds=$(( limit * 60.0 ))
[[ $port == <1024-65535> ]] || { print -u2 "--port takes a number from 1024 to 65535"; exit 1 }
repo=${0:A:h:h:h}
eval "real_home=~$(id -un)"
real_support="$real_home/Library/Application Support/H3cApp"
real_domain=dev.kamahara.h3c-app

# 512x512, 5 s, 20 steps, reuse 2 and the quality mode are the form's own
# defaults, and the compute mode not named is the Mac's own; a job names
# only what it changes. 3 s is what the jobs that are about a feature, not a
# size, run at. A value that starts with $ is filled in when the job runs.
BASE='{"prompt":"A cat playing with a ball of yarn.","size_profile":"square-medium","seconds":5,"steps":20,"reuse":2,"seed":7}'
JOBS=$(jq -c . <<'EOF'
[
 {"name": "std_resident", "label": "標準設定・常駐", "change": {"compute_mode": "resident"}, "how": {}},
 {"name": "std_cache", "label": "標準設定・int8キャッシュ", "change": {"compute_mode": "attentionCache"}, "how": {}},
 {"name": "std_ssd", "label": "標準設定・SSD streaming", "change": {"compute_mode": "ssdStreaming"}, "how": {}},
 {"name": "fast", "label": "速度モード：高速", "change": {"speed_mode": "fast"}, "how": {}},
 {"name": "fastest", "label": "速度モード：最速", "change": {"speed_mode": "fastest"}, "how": {}},
 {"name": "fast_attention", "label": "高速モード（CCV）", "change": {"fast_attention": true}, "how": {}},
 {"name": "size_small", "label": "正方形・小（256×256）", "change": {"size_profile": "square-small"}, "how": {}},
 {"name": "size_landscape", "label": "横長・中（672×384）", "change": {"size_profile": "landscape-medium"}, "how": {}},
 {"name": "size_portrait", "label": "縦長・中（384×672）", "change": {"size_profile": "portrait-medium"}, "how": {}},
 {"name": "size_large", "label": "横長・大（960×544）", "change": {"size_profile": "landscape-large"}, "how": {}},
 {"name": "length_10s", "label": "10秒", "change": {"seconds": 10}, "how": {}},
 {"name": "size_x2", "label": "2倍に拡大（512→1024）、3秒", "change": {"size_profile": "square-medium-x2", "seconds": 3}, "how": {}},
 {"name": "base_3s", "label": "3秒（以下の比較の基準）", "change": {"seconds": 3}, "how": {}},
 {"name": "ssd_3s", "label": "SSD streaming、3秒", "change": {"compute_mode": "ssdStreaming", "seconds": 3}, "how": {}},
 {"name": "reuse_1", "label": "再利用なし（reuse 1）、3秒", "change": {"reuse": 1, "seconds": 3}, "how": {}},
 {"name": "reuse_3", "label": "reuse 3、3秒", "change": {"reuse": 3, "seconds": 3}, "how": {}},
 {"name": "layers_40", "label": "DiT 40層、3秒", "change": {"dit_layers": 40, "seconds": 3}, "how": {}},
 {"name": "first_frame", "label": "最初の画像、3秒", "change": {"first_frame_path": "$FIRST", "seconds": 3}, "how": {}},
 {"name": "first_last", "label": "最初と最後の画像、3秒",
  "change": {"first_frame_path": "$FIRST", "last_frame_path": "$LAST", "seconds": 3}, "how": {}},
 {"name": "reference", "label": "参照画像（Ref2VA）、3秒", "change": {"reference_paths": ["$FIRST"], "seconds": 3}, "how": {}},
 {"name": "refmod_make", "label": "RefModを作る（参照画像から）", "change": null, "how": {"refmod": true}},
 {"name": "refmod_use", "label": "RefModを参照に使う、3秒",
  "change": {"reference_paths": [{"path": "$REFMOD"}], "seconds": 3}, "how": {}},
 {"name": "forced_audio", "label": "音声に合わせる、3秒", "change": {"forced_audio_path": "$AUDIO", "seconds": 3}, "how": {}},
 {"name": "continue", "label": "続きを作る（base_3sの続き）、3秒", "change": {"continue_from": "$VIDEO:base_3s", "seconds": 3}, "how": {}},
 {"name": "chained", "label": "つなげて生成（256×256、16秒、5秒ずつ）",
  "change": {"size_profile": "square-small", "seconds": 16, "chain_clip_seconds": 5}, "how": {}},
 {"name": "count_2", "label": "2本続けて生成、3秒", "change": {"count": 2, "seconds": 3}, "how": {}},
 {"name": "lora_turbo", "label": "Turbo LoRA（4ステップ）＋最初の画像、3秒",
  "change": {"loras": ["$LORA:turbo_4step"], "steps": 4, "first_frame_path": "$FIRST", "seconds": 3}, "how": {}},
 {"name": "lora_dmad8", "label": "DMAD LoRA（8ステップ）、3秒",
  "change": {"loras": ["$LORA:dmad_minimax_h3_8step"], "steps": 8, "seconds": 3}, "how": {}},
 {"name": "pi_h3_4step", "label": "pi-H3（4ステップ、画素ヘッド）、3秒",
  "change": {"loras": ["$LORA:pi_h3_4step_dwd"], "steps": 4, "reuse": 1, "compute_mode": "attentionCache", "seconds": 3}, "how": {}},
 {"name": "pi_h3_2step", "label": "pi-H3（2ステップ、画素ヘッド）、3秒",
  "change": {"loras": ["$LORA:pi_h3_2step_dwd"], "steps": 2, "reuse": 1, "compute_mode": "attentionCache", "seconds": 3}, "how": {}},
 {"name": "pi_h3_fastest", "label": "pi-H3（4ステップ）＋速度モード：最速（トークン削減は効かない）、3秒",
  "change": {"loras": ["$LORA:pi_h3_4step_dwd"], "steps": 4, "speed_mode": "fastest", "compute_mode": "attentionCache", "seconds": 3},
  "how": {}},
 {"name": "pi_h3_first_frame", "label": "pi-H3（4ステップ）＋最初の画像、3秒",
  "change": {"loras": ["$LORA:pi_h3_4step_dwd"], "steps": 4, "reuse": 1, "compute_mode": "attentionCache",
             "first_frame_path": "$FIRST", "seconds": 3}, "how": {}},
 {"name": "pi_h3_first_last", "label": "pi-H3（4ステップ）＋最初と最後の画像、3秒",
  "change": {"loras": ["$LORA:pi_h3_4step_dwd"], "steps": 4, "reuse": 1, "compute_mode": "attentionCache",
             "first_frame_path": "$FIRST", "last_frame_path": "$LAST", "seconds": 3}, "how": {}},
 {"name": "pi_h3_reference", "label": "pi-H3（4ステップ）＋参照画像（Ref2VA）、3秒",
  "change": {"loras": ["$LORA:pi_h3_4step_dwd"], "steps": 4, "reuse": 1, "compute_mode": "attentionCache",
             "reference_paths": ["$FIRST"], "seconds": 3}, "how": {}},
 {"name": "pi_h3_continue", "label": "pi-H3（4ステップ）で続きを作る（pi_h3_4stepの続き）、3秒",
  "change": {"loras": ["$LORA:pi_h3_4step_dwd"], "steps": 4, "reuse": 1, "compute_mode": "attentionCache",
             "continue_from": "$VIDEO:pi_h3_4step", "seconds": 3}, "how": {}},
 {"name": "pi_h3_preview", "label": "pi-H3（4ステップ）＋途中の様子を表示、3秒",
  "change": {"loras": ["$LORA:pi_h3_4step_dwd"], "steps": 4, "reuse": 1, "compute_mode": "attentionCache", "seconds": 3},
  "how": {"preview": true}},
 {"name": "pi_h3_large", "label": "pi-H3（4ステップ）、横長・大（960×544）、5秒",
  "change": {"loras": ["$LORA:pi_h3_4step_dwd"], "steps": 4, "reuse": 1, "compute_mode": "attentionCache",
             "size_profile": "landscape-large"}, "how": {}},
 {"name": "pi_h3_2step_large", "label": "pi-H3（2ステップ）、横長・大（960×544）、5秒",
  "change": {"loras": ["$LORA:pi_h3_2step_dwd"], "steps": 2, "reuse": 1, "compute_mode": "attentionCache",
             "size_profile": "landscape-large"}, "how": {}},
 {"name": "pi_h3_vae_4step", "label": "pi-H3（4ステップ、画素ヘッドなしの版、VAEで復号）、3秒",
  "change": {"loras": ["$LORA:pi_h3_4step_adapter"], "steps": 4, "reuse": 1, "compute_mode": "attentionCache", "seconds": 3}, "how": {}},
 {"name": "pi_h3_vae_2step", "label": "pi-H3（2ステップ、画素ヘッドなしの版、VAEで復号）、3秒",
  "change": {"loras": ["$LORA:pi_h3_2step_adapter"], "steps": 2, "reuse": 1, "compute_mode": "attentionCache", "seconds": 3}, "how": {}},
 {"name": "pi_h3_ssd", "label": "pi-H3（4ステップ）、SSD streaming、3秒",
  "change": {"loras": ["$LORA:pi_h3_4step_dwd"], "steps": 4, "reuse": 1, "compute_mode": "ssdStreaming", "seconds": 3}, "how": {}},
 {"name": "pi_h3_steps_mismatch", "label": "pi-H3（4ステップ版）に2ステップを指定（始める前に断られる）",
  "change": {"loras": ["$LORA:pi_h3_4step_dwd"], "steps": 2, "reuse": 1, "compute_mode": "attentionCache", "seconds": 3},
  "how": {"declined": "pi-H3"}},
 {"name": "pi_h3_steps_refused", "label": "pi-H3（4ステップ版）に8ステップを指定（始める前に断られる）",
  "change": {"loras": ["$LORA:pi_h3_4step_dwd"], "steps": 8, "reuse": 1, "compute_mode": "attentionCache", "seconds": 3},
  "how": {"declined": "pi-H3"}},
 {"name": "hyperflow", "label": "HyperFlow H3 Flash（別のモデル、6ステップ、reuse 1）、3秒",
  "change": {"steps": 6, "reuse": 1, "seconds": 3}, "how": {"hyperflow": true}},
 {"name": "preview", "label": "途中の様子を表示、3秒", "change": {"seconds": 3}, "how": {"preview": true}},
 {"name": "cancel", "label": "生成の途中で中止", "change": {"seconds": 3}, "how": {"cancel": true}},
 {"name": "after_cancel", "label": "中止の後にもう一度生成、3秒", "change": {"seconds": 3}, "how": {}}
]
EOF
)
QUICK=(size_small base_3s ssd_3s first_frame reference cancel after_cancel)
[[ -n $JOBS ]] || { print -u2 "this needs jq, which macOS has from version 15"; exit 1 }
if (( list )); then
    jq -r --args '.[] | [(if .name as $n | $ARGS.positional | index($n) then "quick" else "     " end), .name, "|", .label,
                         "|", (.change | tojson), (if .how == {} then "" else (.how | tojson) end)] | join(" ")' \
        $QUICK <<<"$JOBS"
    exit 0
fi
(( quick && $#only == 0 )) && only=($QUICK)
if (( $#arguments < 2 )); then
    sed -n '2,/^emulate/p' "$0" | sed -e '$d' -e 's/^# \{0,1\}//' >&2
    exit 1
fi

source_path=${arguments[1]:A} work=${arguments[2]:A}
bundle=
[[ -d $source_path ]] && bundle=$source_path
home=$work/home
support="$home/Library/Application Support/H3cApp"
appdir=$work/app
app=$appdir/H3cAppDevCheck
inputs=$work/inputs
log_path=$work/app.log
results_path=$work/results.jsonl
address=http://127.0.0.1:$port
app_pid=
main_id= main_name=      # the model the jobs run with
hyper_looked=0 hyper_id= hyper_name=
run_info='{}'            # the app and the Mac, put in every record
skip_reason=             # what a job needs that is not on this Mac
record=


die() {
    print -u2 -r -- "$*"
    exit 1
}

now_text() {
    strftime "%Y-%m-%d %H:%M:%S" $EPOCHSECONDS
}

forget_preferences() {
    defaults delete H3cAppDevCheck >/dev/null 2>&1
    rm -f -- "$real_home/Library/Preferences/H3cAppDevCheck.plist"
}

# api <path> [<JSON to post, or nothing to get>] [<timeout>]: the answer's
# status in $code and its text in $body.
api() {
    code=000 body=
    [[ -r $support/api-token ]] || return 1
    local token=$(<"$support/api-token") out=$work/answer
    if [[ -n $2 ]]; then
        code=$(curl -s -m ${3:-60} -o "$out" -w '%{http_code}' -H "Authorization: Bearer $token" \
                    -H 'Content-Type: application/json' --data-binary "$2" "$address$1")
    else
        code=$(curl -s -m ${3:-60} -o "$out" -w '%{http_code}' -H "Authorization: Bearer $token" "$address$1")
    fi
    [[ $code == <-> && $code != 000 ]] || { code=000; return 1 }
    body=$(<"$out")
    return 0
}

# What /api/status says under a key, or nothing.
status_of() {
    api /api/status || return 1
    jq -r "$1 // empty" <<<"$body" 2>/dev/null
}

alive() {
    [[ -n $app_pid ]] && kill -0 $app_pid 2>/dev/null
}

# The app's physical footprint in GiB (what Activity Monitor calls memory),
# in $REPLY; nothing when it cannot be read.
footprint_gib() {
    local bytes=$(LC_ALL=C footprint -f bytes $app_pid 2>/dev/null |
                  LC_ALL=C sed -n 's/.*Footprint:[[:space:]]*\([0-9][0-9]*\).*/\1/p' | head -1)
    REPLY=
    [[ -n $bytes ]] && REPLY=$(( bytes / 1073741824.0 ))
}

swap_used() {
    local megabytes=$(sysctl -n vm.swapusage | sed -n 's/.*used = \([0-9.]*\)M.*/\1/p')
    [[ -n $megabytes ]] && print $(( megabytes / 1024.0 )) || print null
}

power() {
    local drawing=$(pmset -g batt | head -1)
    drawing=${${drawing#Now drawing from }//\'/}
    print -r -- "${drawing:-?}, low power $(pmset -g | awk '/lowpowermode/{print $2}')"
}

executable() {
    if [[ -z $bundle ]]; then
        print -r -- "$source_path"
        return
    fi
    local name=$(plutil -extract CFBundleExecutable raw -o - "$bundle/Contents/Info.plist" 2>/dev/null)
    print -r -- "$bundle/Contents/MacOS/${name:-H3cApp}"
}

app_version() {
    if [[ -n $bundle ]]; then
        local info=$bundle/Contents/Info.plist
        local version=$(plutil -extract CFBundleShortVersionString raw -o - "$info" 2>/dev/null)
        local build=$(plutil -extract CFBundleVersion raw -o - "$info" 2>/dev/null)
        print -r -- "${version:-?}（${build:-?}）"
        return
    fi
    local commit=$(git -C "$repo" rev-parse --short HEAD 2>/dev/null)
    local changed=$(git -C "$repo" status --porcelain --untracked-files=no 2>/dev/null)
    print -r -- "開発ビルド、コミット ${commit:-?}${changed:+（未コミットの変更あり）}"
}

machine() {
    jq -cn --arg chip "$(sysctl -n machdep.cpu.brand_string)" --arg model "$(sysctl -n hw.model)" \
           --argjson memory "$(( $(sysctl -n hw.memsize) / 1073741824 ))" --arg macos "$(sw_vers -productVersion)" \
           '{chip: $chip, model: $model, memory_gb: $memory, macos: $macos}'
}

set_up() {
    curl -s -m 1 -o /dev/null $address/
    (( $? == 7 )) || die "port $port is in use: quit H3cApp (or whatever listens there) first, or give another with --port"
    local shader folders folder
    if [[ -n $bundle ]]; then
        shader=$bundle/Contents/Resources/h3_shaders.metal folders=$bundle/Contents/Resources
    else
        shader=$repo/build/h3_shaders.metal folders=$repo/native/H3Spike/Packaging
    fi
    [[ -f $(executable) && -f $shader ]] ||
        die "not an H3cApp.app, or a development build without its repository: $source_path"
    mkdir -p "$home/Library/Preferences" "$appdir" "$inputs" || die "cannot make the work dir: $work"
    cp -f -- "$(executable)" "$app" || die "cannot copy the app"
    if [[ -n $bundle ]]; then
        local signed
        signed=$(codesign --force --sign - "$app" 2>&1) || die "could not sign the copy of the app: $signed"
    fi
    cp -f -- "$shader" "$appdir/"
    for folder in Previews Upscalers en.lproj ja.lproj; do
        rm -rf -- "$appdir/$folder"
        [[ -d $folders/$folder ]] && cp -R -- "$folders/$folder" "$appdir/$folder"
    done
    # The installed app's model and LoRA lists, so that the copy sees them
    # where they are; nothing else of its settings. Both are JSON kept as
    # data, which plutil gives and takes as base64.
    local exported=$(defaults export $real_domain - 2>/dev/null)
    local models_data=$(plutil -extract 'H3ModelLibrary\.models' raw -o - - <<<"$exported" 2>/dev/null)
    local models=$(base64 -d <<<"$models_data" 2>/dev/null | jq -c . 2>/dev/null)
    [[ -n $models && $(jq 'length' <<<"$models") != 0 ]] ||
        die "the installed H3cApp has no model: start it once and add one first"
    local active=$(plutil -extract 'H3ModelLibrary\.activeModelID' raw -o - - <<<"$exported" 2>/dev/null)
    active=$(jq -r --arg a "$active" '([.[] | select(.id == $a)][0] // .[0]).id' <<<"$models")
    local loras_data=$(plutil -extract 'H3ModelLibrary\.loras' raw -o - - <<<"$exported" 2>/dev/null)
    [[ -n $loras_data ]] || loras_data=$(print -n '[]' | base64)
    # --lora: an entry as the app writes one for a file it is given, with the
    # step count its name says ("4step"). They go first in the list, in the
    # order given, so that モデル管理 shows them without scrolling.
    local lora steps listed added='[]'
    for lora in $extra_loras; do
        [[ -f $lora ]] || die "no such LoRA file: $lora"
        # diffusers' fixed file name: listed by its folder, as the app does.
        listed=${lora:t}
        [[ ${(L)listed} == diffusion_pytorch_model.* ]] && listed=${lora:h:t}
        steps=$(sed -nE 's/.*[^0-9]([0-9]{1,2})[-_ ]?steps?.*/\1/p' <<<"${(L)listed}")
        added=$(jq -c --arg id "$(uuidgen)" --arg name "$listed" --arg path "$lora" --arg steps "$steps" \
                   '. + [{id: $id, name: $name, path: $path, scaleText: "",
                          recommendedSteps: (if $steps == "" then null else ($steps | tonumber) end),
                          enabled: false, sampler: "auto", audioShiftText: "", formerPaths: []}]' <<<"$added") ||
            die "cannot add the LoRA to the copy's list: $lora"
    done
    if [[ $added != '[]' ]]; then
        loras_data=$(base64 -d <<<"$loras_data" |
            jq -c --argjson added "$added" '($added | map(.path)) as $paths
                                            | $added + [.[] | select(.path as $p | $paths | index($p) | not)]' | base64) ||
            die "cannot add the LoRAs to the copy's list"
    fi
    local own=$work/own.plist
    print '<?xml version="1.0" encoding="UTF-8"?><plist version="1.0"><dict/></plist>' > "$own"
    plutil -insert 'H3cApp\.apiEnabled' -bool YES "$own" &&
        plutil -insert 'H3ModelLibrary\.models' -data "$models_data" "$own" &&
        plutil -insert 'H3ModelLibrary\.activeModelID' -string "$active" "$own" &&
        plutil -insert 'H3ModelLibrary\.loras' -data "$loras_data" "$own" || die "cannot write the copy's preferences"
    forget_preferences
    defaults import H3cAppDevCheck "$own" || die "cannot set the copy's preferences"
    rm -f -- "$own"
    local id entry
    for id in ${(f)"$(jq -r '.[].id' <<<"$models")"}; do
        [[ -d $real_support/cache/$id ]] || continue
        mkdir -p "$support/cache/$id"
        for entry in "$real_support/cache/$id"/*.cache(N); do
            [[ -e $support/cache/$id/${entry:t} || -L $support/cache/$id/${entry:t} ]] ||
                ln -s "$entry" "$support/cache/$id/${entry:t}"
        done
    done
    if [[ -f $real_support/Preview/taeh3.safetensors && ! -e $support/Preview/taeh3.safetensors ]]; then
        mkdir -p "$support/Preview"
        cp -- "$real_support/Preview/taeh3.safetensors" "$support/Preview/"
    fi
}

# The family of a model (h3 or hyperflow) in $REPLY.
family() {
    REPLY=
    api "/api/models/detail?id=$1" '' 300 && [[ $code == 200 ]] && REPLY=$(jq -r '.family // empty' <<<"$body" 2>/dev/null)
}

# Makes the model the app's and waits for it; whether it loaded.
select_model() {
    api /api/models
    if [[ $(jq -r '[.[] | select(.active)][0].id // empty' <<<"$body" 2>/dev/null) != $1 ]]; then
        api /api/models/select "$(jq -cn --arg id "$1" '{id: $id}')"
        sleep 2
    fi
    local -F deadline=$(( EPOCHREALTIME + 900 ))
    while (( EPOCHREALTIME < deadline )); do
        [[ $(status_of .engine_state) == ready ]] && return 0
        sleep 0.5
    done
    return 1
}

start() {
    rm -f -- "$support/api-token"
    ( cd "$appdir" && CFFIXED_USER_HOME=$home H3_MEMORY_TRACE=1 H3_API_PORT=$port exec "$app" >> "$log_path" 2>&1 ) &
    app_pid=$!
    local -F deadline=$(( EPOCHREALTIME + 60 ))
    code=000
    while (( EPOCHREALTIME < deadline )) && alive; do
        api /api/status && [[ $code == 200 ]] && break
        sleep 0.5
    done
    alive && [[ $code == 200 ]] || die "the app did not answer on $address (see $log_path)"
    if [[ -z $main_id ]]; then
        api /api/models
        local models=$body id
        main_id=$(jq -r '([.[] | select(.active)][0] // .[0]).id' <<<"$models")
        family $main_id
        if [[ $REPLY == hyperflow ]]; then   # it takes its own steps and reuse: not the jobs' model
            local found=
            for id in ${(f)"$(jq -r --arg m "$main_id" '.[] | select(.id != $m) | .id' <<<"$models")"}; do
                family $id
                [[ $REPLY != hyperflow ]] && { found=$id; break }
            done
            [[ -n $found ]] || die "the installed H3cApp has no MiniMax-H3 model, only HyperFlow H3 Flash"
            main_id=$found
        fi
        main_name=$(jq -r --arg m "$main_id" '.[] | select(.id == $m) | .name' <<<"$models")
    fi
    select_model $main_id || die "the model did not load: $(status_of .engine_state) (see $log_path)"
}

stop() {
    if alive; then
        kill -TERM $app_pid 2>/dev/null
        local tick
        for tick in {1..40}; do
            alive || break
            sleep 0.5
        done
        alive && kill -KILL $app_pid 2>/dev/null
        wait $app_pid 2>/dev/null
    fi
    sleep 1
    forget_preferences
    rm -f -- "$work/answer"
}

project_folder() {
    api /api/project && jq -r '.path // empty' <<<"$body" 2>/dev/null
}


# --- Reading MP4 and WAV files: bytes as hex, with od ---

# hex <file> <offset> <count>
hex() {
    od -An -v -tx1 -j $2 -N $3 -- "$1" 2>/dev/null | tr -d ' \n'
}

# A little-endian number given as hex, in $REPLY.
little() {
    local digits=$1 turned=
    while (( $#digits )); do
        turned=${digits[1,2]}$turned
        digits=${digits[3,-1]}
    done
    REPLY=$(( 16#${turned:-0} ))
}

typeset -A FOURCC=(moov 6d6f6f76 trak 7472616b mdia 6d646961 mdhd 6d646864 hdlr 68646c72 minf 6d696e66
                   stbl 7374626c stsd 73747364 stsz 7374737a vide 76696465 soun 736f756e)

# The boxes of an MP4 file between two offsets, one a line: its type (as
# hex), where its content starts, where it ends.
boxes() {
    local file=$1 head
    integer start=$2 end=$3 size header
    while (( start + 8 <= end )); do
        head=$(hex "$file" $start 16)
        (( $#head >= 16 )) || return
        size=$(( 16#${head[1,8]} )) header=8
        if (( size == 1 )); then
            (( $#head == 32 )) || return
            size=$(( 16#${head[17,32]} )) header=16
        elif (( size == 0 )); then
            size=$(( end - start ))
        fi
        (( size >= header && start + size <= end )) || return
        print ${head[9,16]} $(( start + header )) $(( start + size ))
        (( start += size ))
    done
}

# box <file> <start> <end> <type>...: where the content of the box at that
# path starts and ends, in $reply; fails when it is not there.
box() {
    local file=$1 kind line
    integer from=$2 to=$3
    shift 3
    for kind in "$@"; do
        line=$(boxes "$file" $from $to | grep -m1 "^$FOURCC[$kind] ") || return 1
        from=${${(s: :)line}[2]} to=${${(s: :)line}[3]}
    done
    reply=($from $to)
}

# What an MP4 file holds, read from its own tables (no decoder is run), as
# a JSON object.
probe() {
    local file=$1 kind head handler codec name audio= video=
    integer bytes from to header scale length entry width height frames bad=0
    bytes=$(stat -f %z -- "$file" 2>/dev/null)
    if box "$file" 0 $bytes moov; then
        boxes "$file" $reply | while read -r kind from to; do
            [[ $kind == $FOURCC[trak] ]] || continue
            box "$file" $from $to mdia mdhd || { bad=1; break }
            head=$(hex "$file" $reply[1] 32)
            (( $#head == 64 )) || { bad=1; break }
            if [[ ${head[1,2]} == 01 ]]; then
                scale=$(( 16#${head[41,48]} )) length=$(( 16#${head[49,64]} ))
            else
                scale=$(( 16#${head[25,32]} )) length=$(( 16#${head[33,40]} ))
            fi
            box "$file" $from $to mdia hdlr || { bad=1; break }
            handler=$(hex "$file" $(( reply[1] + 8 )) 4)
            box "$file" $from $to mdia minf stbl || { bad=1; break }
            from=$reply[1] to=$reply[2]
            box "$file" $from $to stsd || { bad=1; break }
            head=$(hex "$file" $reply[1] 44)
            (( $#head == 88 )) || { bad=1; break }
            name=${head[25,32]}
            codec=${$(print -n "\x${name[1,2]}\x${name[3,4]}\x${name[5,6]}\x${name[7,8]}")// /}
            case $codec in
                hvc1|hev1) codec=hevc ;;
                avc1) codec=h264 ;;
                mp4a) codec=aac ;;
            esac
            if [[ $handler == $FOURCC[vide] ]]; then
                (( scale > 0 )) || { bad=1; break }
                width=$(( 16#${head[81,84]} )) height=$(( 16#${head[85,88]} ))
                box "$file" $from $to stsz || { bad=1; break }
                head=$(hex "$file" $(( reply[1] + 8 )) 4)
                (( $#head == 8 )) || { bad=1; break }
                frames=$(( 16#$head ))
                video=$(jq -cn --argjson w $width --argjson h $height --arg codec "$codec" --argjson frames $frames \
                           --argjson length $length --argjson scale $scale \
                           '{width: $w, height: $h, codec: $codec, frames: $frames, seconds: (($length / $scale * 100 | round) / 100)}')
            elif [[ $handler == $FOURCC[soun] ]]; then
                audio=$codec
            fi
        done
    else
        bad=1
    fi
    if (( bad || frames == 0 )) || [[ -z $video ]]; then
        print '{"unreadable":true}'
    else
        jq -c --argjson bytes $bytes --arg audio "$audio" \
            '{bytes: $bytes} + . + (if $audio != "" then {audio: $audio} else {} end)' <<<"$video"
    fi
}

# The first seconds of a 16-bit PCM WAV file, as a plain one: afconvert
# writes the extensible header, which not everything reads.
wav_head() {
    local file=$1 target=$2 head kind text
    integer seconds=$3 position=12 total size channels=0 rate=0 block=0 bits=0 at=-1 samples=0 count value
    total=$(stat -f %z -- "$file" 2>/dev/null) || return 1
    while (( position + 8 <= total )); do
        head=$(hex "$file" $position 8)
        (( $#head == 16 )) || break
        kind=${head[1,8]}
        little ${head[9,16]}; size=$REPLY
        if [[ $kind == 666d7420 ]]; then      # "fmt "
            head=$(hex "$file" $(( position + 8 )) 16)
            (( $#head == 32 )) || return 1
            little ${head[5,8]}; channels=$REPLY
            little ${head[9,16]}; rate=$REPLY
            little ${head[25,28]}; block=$REPLY
            little ${head[29,32]}; bits=$REPLY
        elif [[ $kind == 64617461 ]]; then    # "data"
            at=$(( position + 8 )) samples=$size
            (( at + samples > total )) && samples=$(( total - at ))
        fi
        (( position += 8 + size + (size & 1) ))
    done
    (( channels && rate && block && bits && at >= 0 )) || return 1
    count=$(( seconds * rate * block ))
    (( count > samples )) && count=$samples
    # The 44 bytes of a plain header, written as \x.. for printf.
    text='RIFF'
    for value in $(( 36 + count )); do printf -v head '\\x%02x\\x%02x\\x%02x\\x%02x' \
        $(( value & 255 )) $(( value >> 8 & 255 )) $(( value >> 16 & 255 )) $(( value >> 24 & 255 )); text+=$head; done
    text+='WAVEfmt \x10\x00\x00\x00\x01\x00'
    printf -v head '\\x%02x\\x%02x' $(( channels & 255 )) $(( channels >> 8 & 255 )); text+=$head
    for value in $rate $(( rate * block )); do printf -v head '\\x%02x\\x%02x\\x%02x\\x%02x' \
        $(( value & 255 )) $(( value >> 8 & 255 )) $(( value >> 16 & 255 )) $(( value >> 24 & 255 )); text+=$head; done
    printf -v head '\\x%02x\\x%02x\\x%02x\\x%02x' $(( block & 255 )) $(( block >> 8 & 255 )) $(( bits & 255 )) $(( bits >> 8 & 255 ))
    text+=$head'data'
    for value in $count; do printf -v head '\\x%02x\\x%02x\\x%02x\\x%02x' \
        $(( value & 255 )) $(( value >> 8 & 255 )) $(( value >> 16 & 255 )) $(( value >> 24 & 255 )); text+=$head; done
    {
        printf "$text"
        tail -c +$(( at + 1 )) -- "$file" | head -c $count
    } > "$target.part" && mv -f -- "$target.part" "$target"
}


# --- The jobs ---

# The records so far as one object, a job's name to its last record.
records() {
    if [[ -s $results_path ]]; then
        jq -cn 'reduce inputs as $r ({}; .[$r.name] = $r)' "$results_path"
    else
        print '{}'
    fi
}

# A JSON value, or a string, under a key of $record.
put() {
    local changed
    changed=$(jq -c --arg k "$1" --argjson v "$2" '.[$k] = $v' <<<"$record") && record=$changed ||
        print -u2 -r -- "    (could not record $1: $2)"
}
puts() {
    local changed
    changed=$(jq -c --arg k "$1" --arg v "$2" '.[$k] = $v' <<<"$record") && record=$changed
}

# A first and a last picture and a sound, out of the first finished 512x512
# video of this run, with what macOS has: the picture is the frame Quick
# Look shows for the video, the last picture is its mirror image and the
# sound is the video's first 3 s. Returns 3, with the reason, when it cannot.
make_inputs() {
    local first=$inputs/first.png last=$inputs/last.png sound=$inputs/sound.wav
    [[ -e $first && -e $last && -e $sound ]] && return 0
    local video=$(jq -r '[.[] | select(.ok and ((.videos // []) | length > 0) and .videos[0].width == 512)
                          | .videos[0].path][0] // empty' <<<"$finished")
    if [[ -z $video ]]; then
        skip_reason="入力に使う512×512の動画が、まだ無い（512×512の生成を先に行う）"
        return 3
    fi
    local scratch=$inputs/scratch
    mkdir -p "$scratch"
    qlmanage -t -s 512 -o "$scratch" "$video" >/dev/null 2>&1
    local thumbnail=$scratch/${video:t}.png whole=$scratch/whole.wav
    afconvert -f WAVE -d LEI16 "$video" "$whole" >/dev/null 2>&1
    if [[ ! -e $thumbnail || ! -e $whole ]]; then
        skip_reason="動画から画像と音声を取り出せなかった（qlmanage、afconvert）"
        return 3
    fi
    sips -f horizontal "$thumbnail" --out "$last" >/dev/null 2>&1
    if ! wav_head "$whole" "$sound" 3; then
        skip_reason="動画の音声を読めなかった（afconvertのWAV）"
        return 3
    fi
    if [[ ! -e $last ]]; then
        skip_reason="最後の画像を作れなかった（sips）"
        return 3
    fi
    mv -f -- "$thumbnail" "$first"
    rm -rf -- "$inputs/scratch"
    return 0
}

# A request with its $NAMES replaced, in $REPLY; returns 3, with the reason,
# when one is not there.
fill() {
    local request=$1 map='{}' name value part
    for name in ${(f)"$(jq -r '[.. | strings | select(startswith("$"))] | unique | .[]' <<<"$request")"}; do
        case $name in
            '$FIRST'|'$LAST'|'$AUDIO')
                make_inputs || return $?
                case $name in
                    '$FIRST') value=$inputs/first.png ;;
                    '$LAST') value=$inputs/last.png ;;
                    '$AUDIO') value=$inputs/sound.wav ;;
                esac ;;
            '$REFMOD')
                value=$(jq -r '.refmod_make.refmod_path // empty' <<<"$finished")
                if [[ -z $value ]]; then
                    skip_reason="RefModが無い（refmod_makeが済んでいない）"
                    return 3
                fi ;;
            '$VIDEO:'*)
                part=${name#*:}
                value=$(jq -r --arg n "$part" '.[$n] | select(.ok and ((.videos // []) | length > 0)) | .videos[0].path' <<<"$finished")
                if [[ -z $value ]]; then
                    skip_reason="もとにする動画が無い（${part}が済んでいない）"
                    return 3
                fi
                value=${value:t}
                # The app opens a new project each time it starts, and
                # continues only a video of the open one.
                if [[ ! -e $(project_folder)/$value ]]; then
                    skip_reason="もとにする動画が、いま開いているプロジェクトに無い（アプリを起動し直した。${part}と一緒に--redoでやり直す）"
                    return 3
                fi ;;
            '$LORA:'*)
                part=${name#*:}
                api /api/loras
                value=$(jq -r --arg p "$part" '[.[] | select(.name | contains($p)) | .name][0] // empty' <<<"$body" 2>/dev/null)
                if [[ -z $value ]]; then
                    skip_reason="LoRA（名前に${part}を含むもの）が、アプリに登録されていない"
                    return 3
                fi ;;
            *) continue ;;
        esac
        map=$(jq -c --arg k "$name" --arg v "$value" '.[$k] = $v' <<<"$map")
    done
    REPLY=$(jq -c --argjson m "$map" 'walk(if type == "string" and $m[.] != null then $m[.] else . end)' <<<"$request")
}

# What the engine's log (on stdin) says: the peak of each stage, the memory
# plan, and the lines that look like trouble.
parse_log() {
    jq -Rnc '[inputs] as $lines
        | {stage_peaks_gib: (reduce ($lines[] | capture("h3: memory: (?<stage>.+?) peaked at (?<gib>[0-9.]+) GiB"; "g")) as $m
                                 ({}; .[$m.stage] = ([.[$m.stage] // 0, ($m.gib | tonumber)] | max))),
           plan: ([$lines[] | capture("h3: plan: .*peak about (?<gib>[0-9.]+) GiB in the (?<stage>.+)")
                   | {gib: (.gib | tonumber), stage: (.stage | sub("^\\s+"; "") | sub("\\s+$"; ""))}][0]),
           log_notes: ([$lines[] | select(test("error|failed|cannot|refus|warning"; "i"))
                        | sub("^\\s+"; "") | sub("\\s+$"; "") | .[0:200]][0:8])}'
}

make_refmod() {
    make_inputs || return $?
    local -F started=$EPOCHREALTIME peak=0 deadline
    api /api/refmods/extract "$(jq -cn --arg path "$inputs/first.png" '{path: $path, name: "appstatus"}')"
    put http $(( 10#$code ))
    if [[ $code != 200 && $code != 202 ]]; then
        puts error "${$(jq -c . <<<"$body" 2>/dev/null):0:300}"
        return 0
    fi
    deadline=$(( EPOCHREALTIME + 10 ))
    while (( EPOCHREALTIME < deadline )) && [[ $(status_of .refmod_extracting) != true ]]; do
        sleep 0.5
    done
    while [[ $(status_of .refmod_extracting) == true ]] && (( EPOCHREALTIME - started < 1800 )); do
        footprint_gib
        [[ -n $REPLY ]] && (( REPLY > peak )) && peak=$REPLY
        sleep 1
    done
    put wall_seconds $(printf '%.1f' $(( EPOCHREALTIME - started )))
    put footprint_peak_gib $(printf '%.2f' $peak)
    api /api/refmods
    local found=$(jq -r '[.[] | select((.name // "") | contains("appstatus"))][-1].path // empty' <<<"$body" 2>/dev/null)
    if [[ -n $found ]]; then
        puts refmod_path "$found"
        put ok true
    else
        local message=$(status_of .message)
        puts error "${message:-no RefMod came out}"
    fi
    return 0
}

# What the app answers when the memory plan refuses a generation (the
# engine's message, or the form's own), and when the int8 cache a compute
# mode needs is not there or not for this GPU.
is_refusal() {
    [[ $1 == *"more than this Mac's"* || $1 == *"このMacのメモリでは生成できない"* ||
       $1 == *"参照が多すぎて生成できません"* || $1 == *"it takes at most"* ]]
}
is_no_cache() {
    [[ $1 == *"int8キャッシュが見つかりません"* || $1 == *"int8キャッシュを使えません"* ]]
}

# run_job <name> <label> <what it changes> <how it is run>: the record, in
# $record. Returns 3, with the reason, when the job is to be skipped.
run_job() {
    local name=$1 label=$2 change=$3 how=$4
    footprint_gib
    record=$(jq -cn --arg name "$name" --arg label "$label" --arg at "$(now_text)" --argjson run "$run_info" \
                --arg power "$(power)" --argjson swap "$(swap_used)" --argjson idle "$(printf '%.2f' ${REPLY:-0})" \
                '{name: $name, label: $label, ok: false, at: $at, run: $run, power: $power,
                  swap_before_gib: $swap, idle_before_gib: $idle}')
    integer offset=$(stat -f %z -- "$log_path")
    if [[ $how == *'"refmod":true'* ]]; then
        make_refmod
        return $?
    fi
    local request
    fill "$(jq -c --argjson base "$BASE" '$base + .' <<<"$change")" || return $?
    request=$REPLY
    puts model "$main_name"
    if [[ $how == *'"hyperflow":true'* ]]; then
        if (( ! hyper_looked )); then
            hyper_looked=1
            api /api/models
            local id models=$body
            for id in ${(f)"$(jq -r --arg m "$main_id" '[.[] | select(.id != $m)]
                    | sort_by(.name | ascii_downcase | contains("hyper") | not) | .[].id' <<<"$models")"}; do
                family $id
                if [[ $REPLY == hyperflow ]]; then
                    hyper_id=$id hyper_name=$(jq -r --arg m "$id" '.[] | select(.id == $m) | .name' <<<"$models")
                    break
                fi
            done
        fi
        if [[ -z $hyper_id ]]; then
            skip_reason="HyperFlow H3 Flashのモデルが、アプリに登録されていない"
            return 3
        fi
        puts model "$hyper_name"
        if ! select_model $hyper_id; then
            puts error "the model did not load: $(status_of .engine_state)"
            select_model $main_id
            return 0
        fi
    fi
    if [[ $how == *'"preview":true'* ]]; then
        api /api/preview/settings '{"enabled":true}'
        put preview_settings "$(jq -c . <<<"$body" 2>/dev/null || print null)"
    fi
    local folder=$(project_folder)
    local -a before after new
    [[ -d $folder ]] && before=("$folder"/*(ND:t))
    local -F started=$EPOCHREALTIME
    api /api/generate "$request"
    if [[ $code == 409 ]] && is_no_cache "$body"; then
        if [[ $(jq 'has("compute_mode")' <<<"$request") == true ]]; then
            skip_reason=$(jq -r '.error // "int8キャッシュが無い"' <<<"$body" 2>/dev/null)
            return 3
        fi
        # The Mac's own mode is the int8 cache and there is none: the mode a
        # user is told to switch to.
        request=$(jq -c '.compute_mode = "ssdStreaming"' <<<"$request")
        puts fallback ssdStreaming
        started=$EPOCHREALTIME
        api /api/generate "$request"
    fi
    put request "$(jq -c 'del(.prompt)' <<<"$request")"
    put http $(( 10#$code ))
    local error= error_message= videos='[]'
    # "declined": the app is to say no before anything starts, in words that
    # hold this text.
    local declined=$(jq -r '.declined // empty' <<<"$how")
    if [[ $code != 202 ]]; then
        error=${$(jq -c . <<<"$body" 2>/dev/null || jq -Rsc '{error: .}' <<<"$body"):0:400}
        if [[ -n $declined && $code == 409 && $body == *"$declined"* ]]; then
            puts declined "$(jq -r '.error // empty' <<<"$body" 2>/dev/null)"
            error=
            put ok true
        fi
    else
        local -a order spent
        local last_phase= phase busy available
        local -F moment last_time=$started phase_started=$started sampled=0 peak=0
        integer seen=0 cancelled=0 unfinished=1 previewed=0 index
        while (( EPOCHREALTIME - started < limit_seconds )); do
            if ! alive; then
                wait $app_pid 2>/dev/null
                error="the app exited with status $?"
                unfinished=0
                break
            fi
            busy=
            api /api/status && IFS=$'\t' read -r busy phase available <<<"$(jq -r '[(.is_generating // false),
                ((.phase // "") | if . == "" then "-" else . end), (.denoise_preview.available // false)] | @tsv' <<<"$body" 2>/dev/null)"
            if [[ -z $busy ]]; then   # the app is busy; ask again
                sleep 1
                continue
            fi
            [[ $busy == true ]] && seen=1
            moment=$EPOCHREALTIME
            if [[ -n $last_phase ]]; then
                index=${order[(Ie)$last_phase]}
                spent[index]=$(( spent[index] + moment - last_time ))
            fi
            if [[ $phase != $last_phase ]]; then
                phase_started=$moment
                (( ${order[(Ie)$phase]} )) || { order+=("$phase"); spent+=(0) }
            fi
            last_phase=$phase last_time=$moment
            if (( moment - sampled >= 1.0 )); then
                sampled=$moment
                footprint_gib
                [[ -n $REPLY ]] && (( REPLY > peak )) && peak=$REPLY
            fi
            if [[ $how == *'"preview":true'* && $available == true ]] && (( ! previewed )); then
                if [[ $(curl -s -m 60 -o "$work/preview_$name.img" -w '%{http_code}' \
                            -H "Authorization: Bearer $(<"$support/api-token")" $address/api/preview) == 200 ]]; then
                    previewed=1
                    put preview_bytes $(stat -f %z -- "$work/preview_$name.img")
                fi
            fi
            if [[ $how == *'"cancel":true'* && $phase == denoise* ]] && (( ! cancelled && moment - phase_started > 15 )); then
                api /api/cancel '{}'
                put cancel_http $(( 10#$code ))
                put cancelled_after_seconds $(printf '%.1f' $(( moment - started )))
                cancelled=1
            fi
            if (( seen )) && [[ $busy != true ]]; then
                unfinished=0
                break
            fi
            if (( ! seen && moment - started > 30 )); then
                error="the generation did not start"
                unfinished=0
                break
            fi
            sleep 0.5
        done
        if (( unfinished )); then
            api /api/cancel '{}'
            local -F deadline=$(( EPOCHREALTIME + 300 ))
            while (( EPOCHREALTIME < deadline )) && [[ $(status_of .is_generating) == true ]]; do
                sleep 0.5
            done
            error="not finished after $(printf '%.0f' $limit) minutes; cancelled"
        fi
        put wall_seconds $(printf '%.1f' $(( EPOCHREALTIME - started )))
        put footprint_peak_gib $(printf '%.2f' $peak)
        put phases "$(for index in {1..$#order}; do (( $#order )) && print -r -- "${order[index]}"$'\t'"${spent[index]}"; done |
                      jq -Rnc '[inputs | split("\t") | select((.[1] | tonumber) >= 0.5) | [.[0], ((.[1] | tonumber) * 10 | round) / 10]]')"
        if alive; then
            error_message=$(status_of .error_message)
            sleep 5
            footprint_gib
            put idle_after_gib $(printf '%.2f' ${REPLY:-0})
        fi
        local now_folder=$(project_folder)
        [[ -n $now_folder ]] && folder=$now_folder
        [[ -d $folder ]] && after=("$folder"/*(ND:t))
        new=(${after:|before})
        new=(${(o)new})
        local entry info changed
        for entry in $new; do
            [[ $entry == *.mp4 ]] || continue
            info=$(probe "$folder/$entry" | jq -c --arg path "$folder/$entry" '.path = $path')
            if [[ -e $folder/${entry%.mp4}.json ]]; then
                changed=$(jq -c --slurpfile made "$folder/${entry%.mp4}.json" '$made[0] as $m
                    | .generation_seconds = ((($m.generationSeconds // 0) * 10 | round) / 10)
                    | .as_made = {compute_mode: $m.draft.computeMode, speed_mode: $m.draft.speedMode,
                                  reuse: $m.effectiveDenoiseReuse, steps: $m.draft.steps,
                                  frames: $m.actualFrameCount, ccv_calls: $m.ccvAttentionCalls}' <<<"$info" 2>/dev/null) &&
                    info=$changed
            fi
            if [[ -e $folder/${entry%.mp4}.h3lt ]]; then
                info=$(jq -c --arg sum "$(openssl dgst -sha256 -r "$folder/${entry%.mp4}.h3lt" | cut -c1-16)" \
                          '.latent_sha256 = $sum' <<<"$info")
            fi
            videos=$(jq -c --argjson video "$info" '. + [$video]' <<<"$videos")
        done
        # The finished video first: the longest, and of those the last made.
        videos=$(jq -c 'sort_by([-(.seconds // 0), .path])' <<<"$videos")
        put videos "$videos"
        if [[ $how == *'"cancel":true'* ]]; then
            (( cancelled )) && [[ $videos == '[]' && -z $error ]] && put ok true
        else
            [[ -n $declined && -z $error ]] && error="the app began a generation it was to decline"
            [[ $(jq '(length > 0) and (.[0].unreadable | not)' <<<"$videos") == true && -z $error && -z $error_message ]] &&
                put ok true
        fi
    fi
    [[ -n $error ]] && puts error "$error"
    [[ -n $error_message ]] && puts error_message "$error_message"
    is_refusal "$error$error_message" && put refused true
    local parsed=$(tail -c +$(( offset + 1 )) -- "$log_path" | parse_log)
    if [[ -n $parsed ]]; then
        put stage_peaks_gib "$(jq -c '.stage_peaks_gib' <<<"$parsed")"
        put log_notes "$(jq -c '.log_notes' <<<"$parsed")"
        [[ $(jq '.plan != null' <<<"$parsed") == true ]] && put plan "$(jq -c '.plan' <<<"$parsed")"
    fi
    put swap_after_gib "$(swap_used)"
    if alive; then
        [[ $how == *'"preview":true'* ]] && api /api/preview/settings '{"enabled":false}'
        [[ $how == *'"hyperflow":true'* ]] && select_model $main_id
    fi
    return 0
}


set_up
trap stop EXIT
trap 'exit 130' INT TERM HUP
run_info=$(jq -cn --arg app "$(app_version)" --argjson machine "$(machine)" '{tool: "app_status.sh", app: $app, machine: $machine}')
start
run_info=$(jq -c --arg device "$(status_of .device)" '.device = (if $device == "" then null else $device end)' <<<"$run_info")
print -r -- "app up: $(jq -r '"\(.app) | \(.device) | \(.machine.memory_gb) GB"' <<<"$run_info") | $(power) | model: $main_name"
integer number=0 total=$(jq 'length' <<<"$JOBS") result
while (( number < total )); do
    finished=$(records)
    IFS=$'\t' read -r name label <<<"$(jq -r --argjson i $number '.[$i] | [.name, .label] | @tsv' <<<"$JOBS")"
    change=$(jq -c --argjson i $number '.[$i].change' <<<"$JOBS")
    how=$(jq -c --argjson i $number '.[$i].how' <<<"$JOBS")
    (( number++ ))
    (( $#only && ! ${only[(Ie)$name]} )) && continue
    if (( ! redo )) && [[ $(jq --arg n "$name" '(.[$n] != null) and (.[$n].skipped | not)' <<<"$finished") == true ]]; then
        continue
    fi
    alive || start
    print -r -- "$(strftime %H:%M:%S $EPOCHSECONDS) $name ..."
    skip_reason=
    run_job "$name" "$label" "$change" "$how"
    result=$?
    if (( result == 3 )); then
        record=$(jq -cn --arg name "$name" --arg label "$label" --arg reason "$skip_reason" --arg at "$(now_text)" \
                    --argjson run "$run_info" --arg power "$(power)" \
                    '{name: $name, label: $label, ok: false, skipped: $reason, at: $at, run: $run, power: $power}')
    elif ! jq -e . <<<"$record" >/dev/null 2>&1; then   # this script's own fault: on to the next job, with that on record
        record=$(jq -cn --arg name "$name" --arg label "$label" --arg at "$(now_text)" --argjson run "$run_info" \
                    '{name: $name, label: $label, ok: false, at: $at, run: $run, error: "the script failed: no record was made"}')
        if alive && [[ $(status_of .is_generating) == true ]]; then
            api /api/cancel '{}'
            while [[ $(status_of .is_generating) == true ]]; do
                sleep 1
            done
        fi
    fi
    print -r -- "$record" >> "$results_path"
    jq -r 'if .skipped then "    skipped: \(.skipped)"
           elif .declined then "    \(if .ok then "ok" else "FAILED" end), declined: \(.declined)"
           else "    \(if .ok then "ok" elif .refused then "REFUSED" else "FAILED" end) \(.wall_seconds) s, peak \(.footprint_peak_gib) GiB \(.error // .error_message // "")"
           end' <<<"$record"
done
print done
