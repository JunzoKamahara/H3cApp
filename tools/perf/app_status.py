"""Runs H3cApp through a list of video generations over its local API and
records, for each, whether it worked, how long it took and how much memory
it used. It needs nothing but macOS, its own python3 (3.9, which comes with
the Command Line Tools) and the app with a model in it:

    python3 app_status.py /Applications/H3cApp.app <work dir> [--quick] [--only=a,b] [--redo] [--limit=minutes] [--list]

or, for a development build (from the repository, which then gives the
shaders and the other resources):

    make libh3.a CCV_DIR=...
    swift build --package-path native/H3Spike --scratch-path "${TMPDIR}H3cApp-build-scratch" -c release --product H3cApp
    python3 tools/perf/app_status.py "${TMPDIR}H3cApp-build-scratch/release/H3cApp" <work dir> ...

    python3 tools/perf/app_status_report.py <work dir> <report.md>

The app under test is a copy of the executable named H3cAppDevCheck, run
with CFFIXED_USER_HOME set to <work dir>/home (as tools/image_phase1 does):
its Application Support, its projects and its preferences (the domain
H3cAppDevCheck, removed before and after) are its own. A copy out of an app
bundle is signed again, ad hoc: the bundle's signature holds only inside the
bundle, and a bundle run in place would use the installed app's settings.
From the installed app it takes, and only reads: the model list, the LoRA
list, the int8 caches (as symbolic links) and the denoise preview's decoder
(a copy). Port 8420 must be free, so quit H3cApp first.

The model is the one selected in the installed app (another when that is a
HyperFlow H3 Flash, which has a job of its own). A job whose model or LoRA
is not there is recorded as skipped, not failed, and so is one that asks for
the int8 cache where there is none; a job that does not name a compute mode
runs with SSD streaming there. --quick is seven short jobs, 25 to 30 minutes
on an M5, for a slow Mac or a first look; --limit is how long one job may
take (60 minutes).

Each job's record is a line of <work dir>/results.jsonl, with the Mac and
the app's version in it; a job already there is skipped unless --redo. The
app is started once and kept, as a user would keep it; if it exits, the job
is recorded as failed and the app is started again. Keep the work dir
outside ~/Documents."""
import hashlib, json, os, plistlib, re, shutil, signal, socket, struct, subprocess, sys, time, wave
import urllib.error, urllib.request

arguments = [a for a in sys.argv[1:] if not a.startswith("--")]
flags = [a for a in sys.argv[1:] if a.startswith("--")]
only = next((f.split("=", 1)[1].split(",") for f in flags if f.startswith("--only=")), None)
limit = 60 * float(next((f.split("=", 1)[1] for f in flags if f.startswith("--limit=")), 60))
redo = "--redo" in flags
repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
real_support = os.path.expanduser("~/Library/Application Support/H3cApp")
real_domain = "dev.kamahara.h3c-app"
prompt = "A cat playing with a ball of yarn."

# 512x512, 5 s, 20 steps, reuse 2 and the quality mode are the form's own
# defaults, and the compute mode not named is the Mac's own; a job names
# only what it changes. `short` is the 3 s the jobs that are about a
# feature, not a size, run at.
base = {"prompt": prompt, "size_profile": "square-medium", "seconds": 5, "steps": 20, "reuse": 2, "seed": 7}
short = {"seconds": 3}
JOBS = [
    # name, what it is, what it changes, how it is run
    ("std_resident", "標準設定・常駐", {"compute_mode": "resident"}, {}),
    ("std_cache", "標準設定・int8キャッシュ", {"compute_mode": "attentionCache"}, {}),
    ("std_ssd", "標準設定・SSD streaming", {"compute_mode": "ssdStreaming"}, {}),
    ("fast", "速度モード：高速", {"speed_mode": "fast"}, {}),
    ("fastest", "速度モード：最速", {"speed_mode": "fastest"}, {}),
    ("fast_attention", "高速モード（CCV）", {"fast_attention": True}, {}),
    ("size_small", "正方形・小（256×256）", {"size_profile": "square-small"}, {}),
    ("size_landscape", "横長・中（672×384）", {"size_profile": "landscape-medium"}, {}),
    ("size_portrait", "縦長・中（384×672）", {"size_profile": "portrait-medium"}, {}),
    ("size_large", "横長・大（960×544）", {"size_profile": "landscape-large"}, {}),
    ("length_10s", "10秒", {"seconds": 10}, {}),
    ("size_x2", "2倍に拡大（512→1024）、3秒", {"size_profile": "square-medium-x2", **short}, {}),
    ("base_3s", "3秒（以下の比較の基準）", short, {}),
    ("ssd_3s", "SSD streaming、3秒", {"compute_mode": "ssdStreaming", **short}, {}),
    ("reuse_1", "再利用なし（reuse 1）、3秒", {"reuse": 1, **short}, {}),
    ("reuse_3", "reuse 3、3秒", {"reuse": 3, **short}, {}),
    ("layers_40", "DiT 40層、3秒", {"dit_layers": 40, **short}, {}),
    ("first_frame", "最初の画像、3秒", {"first_frame_path": "$FIRST", **short}, {}),
    ("first_last", "最初と最後の画像、3秒", {"first_frame_path": "$FIRST", "last_frame_path": "$LAST", **short}, {}),
    ("reference", "参照画像（Ref2VA）、3秒", {"reference_paths": ["$FIRST"], **short}, {}),
    ("refmod_make", "RefModを作る（参照画像から）", None, {"refmod": True}),
    ("refmod_use", "RefModを参照に使う、3秒", {"reference_paths": [{"path": "$REFMOD"}], **short}, {}),
    ("forced_audio", "音声に合わせる、3秒", {"forced_audio_path": "$AUDIO", **short}, {}),
    ("continue", "続きを作る（base_3sの続き）、3秒", {"continue_from": "$VIDEO:base_3s", **short}, {}),
    ("chained", "つなげて生成（256×256、16秒、5秒ずつ）",
     {"size_profile": "square-small", "seconds": 16, "chain_clip_seconds": 5}, {}),
    ("count_2", "2本続けて生成、3秒", {"count": 2, **short}, {}),
    ("lora_turbo", "Turbo LoRA（4ステップ）＋最初の画像、3秒",
     {"loras": ["$LORA:turbo_4step"], "steps": 4, "first_frame_path": "$FIRST", **short}, {}),
    ("lora_dmad8", "DMAD LoRA（8ステップ）、3秒", {"loras": ["$LORA:dmad_minimax_h3_8step"], "steps": 8, **short}, {}),
    ("hyperflow", "HyperFlow H3 Flash（別のモデル、6ステップ、reuse 1）、3秒", {"steps": 6, "reuse": 1, **short},
     {"hyperflow": True}),
    ("preview", "途中の様子を表示、3秒", short, {"preview": True}),
    ("cancel", "生成の途中で中止", short, {"cancel": True}),
    ("after_cancel", "中止の後にもう一度生成、3秒", short, {}),
]
QUICK = ["size_small", "base_3s", "ssd_3s", "first_frame", "reference", "cancel", "after_cancel"]
if "--list" in flags:
    for name, label, change, how in JOBS:
        print("quick" if name in QUICK else "     ", name, "|", label, "|", change, how or "")
    sys.exit(0)
if "--quick" in flags and not only:
    only = QUICK
if len(arguments) < 2:
    sys.exit(__doc__)

source, work = (os.path.realpath(path) for path in arguments[:2])
bundle = source if os.path.isdir(source) else None
home = os.path.join(work, "home")
support = os.path.join(home, "Library", "Application Support", "H3cApp")
appdir = os.path.join(work, "app")
app = os.path.join(appdir, "H3cAppDevCheck")
inputs = os.path.join(work, "inputs")
log_path = os.path.join(work, "app.log")
results_path = os.path.join(work, "results.jsonl")
address = "http://127.0.0.1:8420"
process = None
main_model = None      # the /api/models entry the jobs run with
hyperflow_model = []   # found when first asked for: [entry or None]
run_info = {}          # the app and the Mac, put in every record


class Skip(Exception):
    """What a job needs is not on this Mac; the reason is the message."""


def shell(*command):
    try:
        return subprocess.run(command, capture_output=True, text=True).stdout
    except FileNotFoundError:
        return ""


def forget_preferences():
    subprocess.run(["defaults", "delete", "H3cAppDevCheck"], capture_output=True)
    import pwd
    leftover = os.path.join(pwd.getpwuid(os.getuid()).pw_dir, "Library", "Preferences", "H3cAppDevCheck.plist")
    if os.path.exists(leftover):
        os.remove(leftover)


def api(path, body=None, raw=False, timeout=60):
    token = open(os.path.join(support, "api-token")).read().strip()
    request = urllib.request.Request(address + path, data=None if body is None else json.dumps(body).encode(),
                                     headers={"Authorization": "Bearer " + token})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = response.read()
            return (response.status, data) if raw else (response.status, json.loads(data))
    except urllib.error.HTTPError as error:
        data = error.read()
        try:
            return (error.code, data) if raw else (error.code, json.loads(data or b"{}"))
        except ValueError:
            return error.code, {"error": data.decode(errors="replace")}


def status():
    return api("/api/status")[1]


def wait(condition, seconds, pause=0.5):
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            if condition():
                return True
        except Exception:
            pass
        time.sleep(pause)
    return False


def footprint():
    """The app's physical footprint in GiB (what Activity Monitor calls memory)."""
    text = shell("footprint", "-f", "bytes", str(process.pid))
    match = re.search(r"Footprint:\s+(\d+)", text)
    return int(match.group(1)) / 2 ** 30 if match else None


def swap_used():
    match = re.search(r"used = ([\d.]+)M", shell("sysctl", "-n", "vm.swapusage"))
    return float(match.group(1)) / 1024 if match else None


def power():
    return (shell("pmset", "-g", "batt").splitlines() or ["?"])[0].replace("Now drawing from ", "").strip("'") + \
        ", low power " + shell("sh", "-c", "pmset -g | awk '/lowpowermode/{print $2}'").strip()


def executable():
    if not bundle:
        return source
    info = plistlib.load(open(os.path.join(bundle, "Contents", "Info.plist"), "rb"))
    return os.path.join(bundle, "Contents", "MacOS", info.get("CFBundleExecutable", "H3cApp"))


def app_version():
    if bundle:
        info = plistlib.load(open(os.path.join(bundle, "Contents", "Info.plist"), "rb"))
        return "%s（%s）" % (info.get("CFBundleShortVersionString", "?"), info.get("CFBundleVersion", "?"))
    commit = shell("git", "-C", repo, "rev-parse", "--short", "HEAD").strip() or "?"
    changed = shell("git", "-C", repo, "status", "--porcelain", "--untracked-files=no").strip()
    return "開発ビルド、コミット %s%s" % (commit, "（未コミットの変更あり）" if changed else "")


def machine():
    return {"chip": shell("sysctl", "-n", "machdep.cpu.brand_string").strip(),
            "model": shell("sysctl", "-n", "hw.model").strip(),
            "memory_gb": int(shell("sysctl", "-n", "hw.memsize") or 0) // 2 ** 30,
            "macos": shell("sw_vers", "-productVersion").strip()}


def set_up():
    with socket.socket() as probe_socket:
        probe_socket.settimeout(1)
        if probe_socket.connect_ex(("127.0.0.1", 8420)) == 0:
            sys.exit("port 8420 is in use: quit H3cApp (or whatever listens there) first")
    if bundle:
        shader, folders = (os.path.join(bundle, "Contents", "Resources", "h3_shaders.metal"),
                           os.path.join(bundle, "Contents", "Resources"))
    else:
        shader, folders = (os.path.join(repo, "build", "h3_shaders.metal"),
                           os.path.join(repo, "native", "H3Spike", "Packaging"))
    if not os.path.exists(executable()) or not os.path.exists(shader):
        sys.exit("not an H3cApp.app, or a development build without its repository: " + source)
    os.makedirs(os.path.join(home, "Library", "Preferences"), exist_ok=True)
    os.makedirs(appdir, exist_ok=True)
    os.makedirs(inputs, exist_ok=True)
    shutil.copy(executable(), app)
    if bundle:
        signed = subprocess.run(["codesign", "--force", "--sign", "-", app], capture_output=True, text=True)
        if signed.returncode:
            sys.exit("could not sign the copy of the app: " + signed.stderr.strip())
    shutil.copy(shader, appdir)
    for folder in ("Previews", "Upscalers", "en.lproj", "ja.lproj"):
        target = os.path.join(appdir, folder)
        shutil.rmtree(target, ignore_errors=True)
        if os.path.isdir(os.path.join(folders, folder)):
            shutil.copytree(os.path.join(folders, folder), target)
    # The installed app's model and LoRA lists, so that the copy sees them
    # where they are; nothing else of its settings.
    exported = subprocess.run(["defaults", "export", real_domain, "-"], capture_output=True).stdout
    real = plistlib.loads(exported) if exported else {}
    models = json.loads(real.get("H3ModelLibrary.models", b"[]"))
    if not models:
        sys.exit("the installed H3cApp has no model: start it once and add one first")
    active = real.get("H3ModelLibrary.activeModelID")
    own = {"H3cApp.apiEnabled": True,
           "H3ModelLibrary.models": json.dumps(models).encode(),
           "H3ModelLibrary.activeModelID": active if any(m["id"] == active for m in models) else models[0]["id"],
           "H3ModelLibrary.loras": real.get("H3ModelLibrary.loras", b"[]")}
    forget_preferences()
    subprocess.run(["defaults", "import", "H3cAppDevCheck", "-"], input=plistlib.dumps(own), check=True)
    for model in models:
        source_cache = os.path.join(real_support, "cache", model["id"])
        if not os.path.isdir(source_cache):
            continue
        target = os.path.join(support, "cache", model["id"])
        os.makedirs(target, exist_ok=True)
        for entry in os.listdir(source_cache):
            if entry.endswith(".cache") and not os.path.lexists(os.path.join(target, entry)):
                os.symlink(os.path.join(source_cache, entry), os.path.join(target, entry))
    decoder = os.path.join(real_support, "Preview", "taeh3.safetensors")
    if os.path.exists(decoder) and not os.path.exists(os.path.join(support, "Preview", "taeh3.safetensors")):
        os.makedirs(os.path.join(support, "Preview"), exist_ok=True)
        shutil.copy(decoder, os.path.join(support, "Preview"))


def family(model):
    code, body = api("/api/models/detail?id=" + model["id"], timeout=300)
    return body.get("family") if code == 200 else None


def select(model):
    """Makes `model` the app's and waits for it; whether it loaded."""
    if next((m for m in api("/api/models")[1] if m.get("active")), {}).get("id") != model["id"]:
        api("/api/models/select", {"id": model["id"]})
        time.sleep(2)
    return wait(lambda: status().get("engine_state") == "ready", 900)


def start():
    global process, main_model
    try:
        os.remove(os.path.join(support, "api-token"))
    except FileNotFoundError:
        pass
    environment = dict(os.environ, CFFIXED_USER_HOME=home, H3_MEMORY_TRACE="1")
    process = subprocess.Popen([app], env=environment, cwd=appdir, stdout=open(log_path, "a"),
                               stderr=subprocess.STDOUT)
    if not wait(lambda: process.poll() is not None or api("/api/status")[0] == 200, 60) or process.poll() is not None:
        sys.exit("the app did not answer on " + address + " (see " + log_path + ")")
    if main_model is None:
        models = api("/api/models")[1]
        main_model = next((m for m in models if m.get("active")), models[0])
        if family(main_model) == "hyperflow":   # it takes its own steps and reuse: not the jobs' model
            main_model = next((m for m in models if m["id"] != main_model["id"] and family(m) != "hyperflow"), None)
            if main_model is None:
                sys.exit("the installed H3cApp has no MiniMax-H3 model, only HyperFlow H3 Flash")
    if not select(main_model):
        sys.exit("the model did not load: %s (see %s)" % (status().get("engine_state"), log_path))


def stop(*_):
    if process and process.poll() is None:
        process.send_signal(signal.SIGTERM)
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    time.sleep(1)
    forget_preferences()


def project_folder():
    return api("/api/project")[1].get("path")


def boxes(data, start, end):
    """The boxes of an MP4 file between two offsets: (type, where its content starts, where it ends)."""
    while start + 8 <= end:
        size, kind = struct.unpack(">I4s", data[start:start + 8])
        header = 8
        if size == 1:
            size, header = struct.unpack(">Q", data[start + 8:start + 16])[0], 16
        elif size == 0:
            size = end - start
        if size < header or start + size > end:
            return
        yield kind, start + header, start + size
        start += size


def box(data, span, *path):
    for wanted in path:
        span = next(((a, b) for kind, a, b in boxes(data, *span) if kind == wanted), None)
        if span is None:
            return None
    return span


def probe(path):
    """What an MP4 file holds, read from its own tables (no decoder is run)."""
    out = {"bytes": os.path.getsize(path)}
    try:
        data = open(path, "rb").read()
        for kind, a, b in boxes(data, *box(data, (0, len(data)), b"moov")):
            if kind != b"trak":
                continue
            header = box(data, (a, b), b"mdia", b"mdhd")[0]
            wide = data[header] == 1
            scale, length = struct.unpack(">IQ", data[header + 20:header + 32]) if wide else \
                struct.unpack(">II", data[header + 12:header + 20])
            handler = data[box(data, (a, b), b"mdia", b"hdlr")[0] + 8:][:4]
            table = box(data, (a, b), b"mdia", b"minf", b"stbl")
            entry = box(data, table, b"stsd")[0]
            codec = data[entry + 12:entry + 16].decode("latin-1").strip()
            codec = {"hvc1": "hevc", "hev1": "hevc", "avc1": "h264", "mp4a": "aac"}.get(codec, codec)
            if handler == b"vide":
                width, height = struct.unpack(">HH", data[entry + 40:entry + 44])
                frames, = struct.unpack(">I", data[box(data, table, b"stsz")[0] + 8:][:4])
                out.update(width=width, height=height, codec=codec, frames=frames, seconds=round(length / scale, 2))
            elif handler == b"soun":
                out["audio"] = codec
    except (OSError, TypeError, ValueError, IndexError, struct.error, ZeroDivisionError):
        return {"unreadable": True}
    return out if out.get("frames") else {"unreadable": True}


def records():
    found = {}
    if os.path.exists(results_path):
        for line in open(results_path):
            record = json.loads(line)
            found[record["name"]] = record
    return found


def wav_head(path, target, seconds):
    """The first seconds of a 16-bit PCM WAV file, as a plain one: afconvert
    writes the extensible header, which the wave module does not read."""
    data = open(path, "rb").read()
    position, form, samples = 12, None, b""
    while position + 8 <= len(data):
        kind, size = struct.unpack("<4sI", data[position:position + 8])
        if kind == b"fmt ":
            form = struct.unpack("<HHIIHH", data[position + 8:position + 24])
        elif kind == b"data":
            samples = data[position + 8:position + 8 + size]
        position += 8 + size + (size & 1)
    _, channels, rate, _, block, bits = form
    with wave.open(target, "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(bits // 8)
        writer.setframerate(rate)
        writer.writeframes(samples[:seconds * rate * block])


def make_inputs(done):
    """A first and a last picture and a sound, out of the first finished
    512x512 video of this run, with what macOS has: the picture is the frame
    Quick Look shows for the video, the last picture is its mirror image and
    the sound is the video's first 3 s."""
    first, last, sound = (os.path.join(inputs, name) for name in ("first.png", "last.png", "sound.wav"))
    if all(os.path.exists(path) for path in (first, last, sound)):
        return
    source_video = next((r["videos"][0]["path"] for r in done.values()
                         if r.get("ok") and r.get("videos") and r["videos"][0].get("width") == 512), None)
    if not source_video:
        raise Skip("入力に使う512×512の動画が、まだ無い（512×512の生成を先に行う）")
    scratch = os.path.join(inputs, "scratch")
    os.makedirs(scratch, exist_ok=True)
    subprocess.run(["qlmanage", "-t", "-s", "512", "-o", scratch, source_video], capture_output=True)
    thumbnail = os.path.join(scratch, os.path.basename(source_video) + ".png")
    whole = os.path.join(scratch, "whole.wav")
    subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16", source_video, whole], capture_output=True)
    if not os.path.exists(thumbnail) or not os.path.exists(whole):
        raise Skip("動画から画像と音声を取り出せなかった（qlmanage、afconvert）")
    subprocess.run(["sips", "-f", "horizontal", thumbnail, "--out", last], capture_output=True)
    try:
        wav_head(whole, sound, 3)
    except (TypeError, ValueError, struct.error, wave.Error):
        raise Skip("動画の音声を読めなかった（afconvertのWAV）")
    if not os.path.exists(last):
        raise Skip("最後の画像を作れなかった（sips）")
    shutil.move(thumbnail, first)
    shutil.rmtree(scratch, ignore_errors=True)


def fill(value, done):
    """A request with its $NAMES replaced; Skip when one is not there."""
    if isinstance(value, dict):
        return {key: fill(item, done) for key, item in value.items()}
    if isinstance(value, list):
        return [fill(item, done) for item in value]
    if not isinstance(value, str) or not value.startswith("$"):
        return value
    if value in ("$FIRST", "$LAST", "$AUDIO"):
        make_inputs(done)
        return os.path.join(inputs, {"$FIRST": "first.png", "$LAST": "last.png", "$AUDIO": "sound.wav"}[value])
    if value == "$REFMOD":
        if not done.get("refmod_make", {}).get("refmod_path"):
            raise Skip("RefModが無い（refmod_makeが済んでいない）")
        return done["refmod_make"]["refmod_path"]
    if value.startswith("$VIDEO:"):
        earlier = done.get(value.split(":", 1)[1], {})
        if not earlier.get("ok") or not earlier.get("videos"):
            raise Skip("もとにする動画が無い（%sが済んでいない）" % value.split(":", 1)[1])
        return os.path.basename(earlier["videos"][0]["path"])
    if value.startswith("$LORA:"):
        part = value.split(":", 1)[1]
        name = next((l["name"] for l in api("/api/loras")[1] if part in l["name"]), None)
        if name is None:
            raise Skip("LoRA（名前に%sを含むもの）が、アプリに登録されていない" % part)
        return name
    return value


def parse_log(text):
    stages = {}
    for phase, peak in re.findall(r"h3: memory: (.+?) peaked at ([\d.]+) GiB", text):
        stages[phase] = max(stages.get(phase, 0.0), float(peak))
    plans = re.findall(r"h3: plan: .*peak about ([\d.]+) GiB in the (.+)", text)
    notes = [line.strip()[:200] for line in text.splitlines()
             if re.search(r"error|failed|cannot|refus|warning", line, re.I)]
    return stages, [{"gib": float(g), "stage": s.strip()} for g, s in plans[:1]], notes[:8]


def make_refmod(record, done):
    make_inputs(done)
    started = time.time()
    code, body = api("/api/refmods/extract", {"path": os.path.join(inputs, "first.png"), "name": "appstatus"})
    record["http"] = code
    if code not in (200, 202):
        record["error"] = json.dumps(body, ensure_ascii=False)[:300]
        return
    peak = 0.0
    wait(lambda: status().get("refmod_extracting"), 10)
    while status().get("refmod_extracting") and time.time() - started < 1800:
        peak = max(peak, footprint() or 0.0)
        time.sleep(1)
    record["wall_seconds"] = round(time.time() - started, 1)
    record["footprint_peak_gib"] = round(peak, 2)
    found = [r for r in api("/api/refmods")[1] if "appstatus" in r.get("name", "")]
    if found:
        record["refmod_path"] = found[-1]["path"]
        record["ok"] = True
    else:
        record["error"] = status().get("message") or "no RefMod came out"


# What the app answers when the memory plan refuses a generation (the
# engine's message, or the form's own), and when the int8 cache a compute
# mode needs is not there or not for this GPU.
REFUSED = re.compile(r"more than this Mac's|このMacのメモリでは生成できない|参照が多すぎて生成できません|it takes at most")
NO_CACHE = re.compile(r"int8キャッシュが見つかりません|int8キャッシュを使えません")


def run(name, label, change, how, done):
    record = {"name": name, "label": label, "ok": False, "at": time.strftime("%Y-%m-%d %H:%M:%S"),
              "run": run_info, "power": power(), "swap_before_gib": swap_used(),
              "idle_before_gib": round(footprint() or 0, 2)}
    offset = os.path.getsize(log_path)
    if how.get("refmod"):
        make_refmod(record, done)
        return record
    request = fill({**base, **change}, done)
    record["model"] = main_model["name"]
    if how.get("hyperflow"):
        if not hyperflow_model:
            others = sorted((m for m in api("/api/models")[1] if m["id"] != main_model["id"]),
                            key=lambda m: "hyper" not in m["name"].lower())
            hyperflow_model.append(next((m for m in others if family(m) == "hyperflow"), None))
        if hyperflow_model[0] is None:
            raise Skip("HyperFlow H3 Flashのモデルが、アプリに登録されていない")
        record["model"] = hyperflow_model[0]["name"]
        if not select(hyperflow_model[0]):
            record["error"] = "the model did not load: " + str(status().get("engine_state"))
            select(main_model)
            return record
    if how.get("preview"):
        record["preview_settings"] = api("/api/preview/settings", {"enabled": True})[1]
    folder = project_folder()
    before = set(os.listdir(folder)) if folder and os.path.isdir(folder) else set()
    started = time.time()
    code, body = api("/api/generate", request)
    if code == 409 and NO_CACHE.search(json.dumps(body, ensure_ascii=False)):
        if "compute_mode" in request:
            raise Skip(body.get("error", "int8キャッシュが無い"))
        # The Mac's own mode is the int8 cache and there is none: the mode a
        # user is told to switch to.
        request["compute_mode"] = record["fallback"] = "ssdStreaming"
        started = time.time()
        code, body = api("/api/generate", request)
    record["request"] = {k: v for k, v in request.items() if k != "prompt"}
    record["http"] = code
    if code != 202:
        record["error"] = json.dumps(body, ensure_ascii=False)[:400]
    else:
        phases, order, peak, last_phase, last_time = {}, [], 0.0, None, started
        seen, sampled, cancelled, phase_started = False, 0.0, False, started
        while time.time() - started < limit:
            if process.poll() is not None:
                record["error"] = "the app exited with status %s" % process.returncode
                break
            try:
                now = status()
            except Exception as error:   # the app is busy; ask again
                time.sleep(1)
                continue
            busy = now.get("is_generating")
            seen = seen or busy
            moment = time.time()
            phase = now.get("phase") or "-"
            if last_phase is not None:
                phases[last_phase] = phases.get(last_phase, 0.0) + moment - last_time
            if phase != last_phase:
                phase_started = moment
                if phase not in order:
                    order.append(phase)
            last_phase, last_time = phase, moment
            if moment - sampled >= 1.0:
                sampled = moment
                peak = max(peak, footprint() or 0.0)
            if how.get("preview") and "preview_bytes" not in record and \
                    (now.get("denoise_preview") or {}).get("available"):
                code, data = api("/api/preview", raw=True)
                if code == 200:
                    record["preview_bytes"] = len(data)
                    open(os.path.join(work, "preview_" + name + ".img"), "wb").write(data)
            if how.get("cancel") and not cancelled and phase.startswith("denoise") and moment - phase_started > 15:
                record["cancel_http"] = api("/api/cancel", {})[0]
                record["cancelled_after_seconds"] = round(moment - started, 1)
                cancelled = True
            if seen and not busy:
                break
            if not seen and moment - started > 30:
                record["error"] = "the generation did not start"
                break
            time.sleep(0.5)
        else:
            api("/api/cancel", {})
            wait(lambda: not status().get("is_generating"), 300)
            record["error"] = "not finished after %.0f minutes; cancelled" % (limit / 60)
        record["wall_seconds"] = round(time.time() - started, 1)
        record["footprint_peak_gib"] = round(peak, 2)
        record["phases"] = [[p, round(phases.get(p, 0.0), 1)] for p in order if phases.get(p, 0.0) >= 0.5]
        if process.poll() is None:
            final = status()
            if final.get("error_message"):
                record["error_message"] = final["error_message"]
            time.sleep(5)
            record["idle_after_gib"] = round(footprint() or 0, 2)
        folder = project_folder() or folder
        new = sorted(set(os.listdir(folder)) - before) if folder and os.path.isdir(folder) else []
        videos = []
        for entry in new:
            if entry.endswith(".mp4"):
                info = probe(os.path.join(folder, entry))
                info["path"] = os.path.join(folder, entry)
                sidecar = os.path.join(folder, entry[:-4] + ".json")
                if os.path.exists(sidecar):
                    made = json.load(open(sidecar))
                    info["generation_seconds"] = round(made.get("generationSeconds", 0), 1)
                    draft = made.get("draft", {})
                    info["as_made"] = {"compute_mode": draft.get("computeMode"), "speed_mode": draft.get("speedMode"),
                                       "reuse": made.get("effectiveDenoiseReuse"), "steps": draft.get("steps"),
                                       "frames": made.get("actualFrameCount"),
                                       "ccv_calls": made.get("ccvAttentionCalls")}
                latent = os.path.join(folder, entry[:-4] + ".h3lt")
                if os.path.exists(latent):
                    info["latent_sha256"] = hashlib.sha256(open(latent, "rb").read()).hexdigest()[:16]
                videos.append(info)
        # The finished video first: the longest, and of those the last made.
        videos.sort(key=lambda v: (-(v.get("seconds") or 0), v["path"]))
        record["videos"] = videos
        if how.get("cancel"):
            record["ok"] = cancelled and not videos and "error" not in record
        else:
            record["ok"] = bool(videos) and not videos[0].get("unreadable") and "error" not in record \
                and "error_message" not in record
    if REFUSED.search(record.get("error", "") + record.get("error_message", "")):
        record["refused"] = True
    with open(log_path, errors="replace") as log:
        log.seek(offset)
        record["stage_peaks_gib"], plan, record["log_notes"] = parse_log(log.read())
    if plan:
        record["plan"] = plan[0]
    record["swap_after_gib"] = swap_used()
    if how.get("preview") and process.poll() is None:
        api("/api/preview/settings", {"enabled": False})
    if how.get("hyperflow") and process.poll() is None:
        select(main_model)
    return record


import atexit
set_up()
atexit.register(stop)
run_info.update(app=app_version(), machine=machine())
start()
run_info["device"] = status().get("device")
print("app up:", run_info["app"], "|", run_info["device"], "|", "%d GB" % run_info["machine"]["memory_gb"], "|",
      power(), "| model:", main_model["name"], flush=True)
for name, label, change, how in JOBS:
    done = records()
    if only and name not in only:
        continue
    if name in done and not redo and not done[name].get("skipped"):
        continue
    if process.poll() is not None:
        start()
    print(time.strftime("%H:%M:%S"), name, "...", flush=True)
    try:
        record = run(name, label, change, how, done)
    except Skip as reason:
        record = {"name": name, "label": label, "ok": False, "skipped": str(reason),
                  "at": time.strftime("%Y-%m-%d %H:%M:%S"), "run": run_info, "power": power()}
    except Exception:   # this script's own fault: on to the next job, with what happened on record
        import traceback
        traceback.print_exc()
        record = {"name": name, "label": label, "ok": False, "at": time.strftime("%Y-%m-%d %H:%M:%S"),
                  "run": run_info, "error": "the script failed: " + traceback.format_exc().strip().splitlines()[-1]}
        if process.poll() is None and wait(lambda: status().get("is_generating"), 3):
            api("/api/cancel", {})
            wait(lambda: not status().get("is_generating"), 300)
    with open(results_path, "a") as out:
        out.write(json.dumps(record, ensure_ascii=False) + "\n")
    if record.get("skipped"):
        print("    skipped:", record["skipped"], flush=True)
    else:
        print("   ", "ok" if record["ok"] else "REFUSED" if record.get("refused") else "FAILED",
              record.get("wall_seconds"), "s, peak", record.get("footprint_peak_gib"), "GiB",
              record.get("error") or record.get("error_message") or "", flush=True)
print("done", flush=True)
