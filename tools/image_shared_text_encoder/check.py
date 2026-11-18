"""Check of the text encoder MiniMax H3 Image shares with a video model
(SharedTextEncoder in native/H3Spike/Sources/H3cApp), driven over the app's
local HTTP API. Nothing is generated and no weights are loaded: the video
models are stand-ins holding a few megabytes named like a text encoder, one
in the work dir and one on a small disk image made here (another volume).

usage: check.py <H3cApp binary> <empty work dir> <weights dir>

<weights dir> holds MiniMax H3 Image's files as Comfy-Org publishes them:
diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors and
vae/minimax_h3_image_vae_fp16.safetensors. They are cloned into the work
dir, which must be on their (APFS) volume, and a record of their hashes is
written beside them so that the app does not read all 22 GB to check them.

The app is run as a copy named H3cAppDevCheck with CFFIXED_USER_HOME set to
<work dir>/home (as tools/image_phase1/roundtrip_check.py does): its
Application Support folder is under the work dir and its preferences are a
domain of its own, deleted before and after. Port 8420 must be free (quit
H3cApp first). The setup fetches the model's LICENSE (and, when they are not
in <weights dir>, vae/config.json and tokenizer.json) from Hugging Face."""
import atexit, hashlib, json, os, shutil, signal, subprocess, sys, time, urllib.error, urllib.request

binary, work, weights = (os.path.realpath(path) for path in sys.argv[1:4])
home = os.path.join(work, "home")
support = os.path.join(home, "Library", "Application Support", "H3cApp")
image = os.path.join(support, "ImageModels", "MiniMax-H3-Image")
app = os.path.join(work, "H3cAppDevCheck")
disk_image = os.path.join(work, "other.dmg")
volume = "/Volumes/H3ShareCheck"
base = "http://127.0.0.1:8420"
record_key = "h3c-app.imageModelTextEncoderSources"
failures = []
process = None

REVISION = "892de7cb61df49753bf15145a711154803a355cb"
WEIGHTS = {
    "transformer/minimax_h3_image_pruned_int8_convrot.safetensors":
        ("diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors", 20_979_314_802,
         "83eb140f1ff95c068f8a6886e6d3f3d0d4ab2e07bb2fe6d2ef992a51146ee622"),
    "vae/minimax_h3_image_vae_fp16.safetensors":
        ("vae/minimax_h3_image_vae_fp16.safetensors", 1_168_505_312,
         "d80e9c2e5cfc9844869961ff5a58c9a9e52182df6bfb30b26018dd7f348721af"),
}
SMALL = {
    "vae/config.json": ("vae/config.json", "7fcc21bec549804ff0a04d8246e7f90e679d4f8d4b0a27a252bbd18c30e3e4bb"),
    "tokenizer/tokenizer.json": ("tokenizer.json", "a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7"),
}


def forget_preferences():
    subprocess.run(["defaults", "delete", "H3cAppDevCheck"], capture_output=True)
    import pwd
    leftover = os.path.join(pwd.getpwuid(os.getuid()).pw_dir, "Library", "Preferences", "H3cAppDevCheck.plist")
    if os.path.exists(leftover):
        os.remove(leftover)


def check(condition, what):
    print(("ok   " if condition else "FAIL ") + what, flush=True)
    if not condition:
        failures.append(what)


def api(path, body=None, post=False):
    token = open(os.path.join(support, "api-token")).read().strip()
    data = json.dumps(body or {}).encode() if post or body is not None else None
    request = urllib.request.Request(base + path, data=data, headers={"Authorization": "Bearer " + token})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read() or b"{}")


def wait(condition, seconds=20):
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            if condition():
                return True
        except Exception:
            pass
        time.sleep(0.2)
    return False


def start():
    global process
    try:
        os.remove(os.path.join(support, "api-token"))
    except FileNotFoundError:
        pass
    process = subprocess.Popen([app], env=dict(os.environ, CFFIXED_USER_HOME=home), cwd=work,
                               stdout=open(os.path.join(work, "app.log"), "a"), stderr=subprocess.STDOUT)
    if not wait(lambda: api("/api/status")[0] == 200):
        sys.exit("the app did not answer on " + base + " (see app.log; is H3cApp already running?)")


def stop():
    process.send_signal(signal.SIGKILL)
    process.wait()


def attach():
    subprocess.run(["hdiutil", "attach", "-nobrowse", disk_image], check=True, capture_output=True)


def detach():
    if os.path.exists(volume):
        subprocess.run(["hdiutil", "detach", volume], check=True, capture_output=True)


def finish(*_):
    if process and process.poll() is None:
        stop()
    time.sleep(1)
    try:
        detach()
    except subprocess.CalledProcessError:
        subprocess.run(["hdiutil", "detach", "-force", volume], capture_output=True)
    forget_preferences()


atexit.register(finish)


def video_model(folder, megabytes):
    """A stand-in: FL2VA/text_encoder with one file of that many megabytes."""
    encoder = os.path.join(folder, "FL2VA", "text_encoder")
    os.makedirs(encoder)
    with open(os.path.join(encoder, "model.gguf"), "wb") as f:
        f.write(os.urandom(megabytes << 20))
    return os.path.join(encoder)


def setup_state():
    return next(item for item in api("/api/image-models")[1]["setups"] if item["model"] == "h3")


def run_setup():
    status, answer = api("/api/image-models/setup", {"confirm_license": True})
    check(status == 200, "the setup starts (%s %s)" % (status, answer.get("error", "")))
    wait(lambda: setup_state()["phase"] == "done" or setup_state()["phase"].startswith("failed"), 120)
    return setup_state()


def real(path):
    # The app spells its home as NSHomeDirectory does (/tmp for /private/tmp).
    return path and os.path.realpath(path)


def dependents(model_id):
    status, answer = api("/api/models/remove", {"id": model_id, "delete_files": True, "dry_run": True})
    return [dict(item, path=real(item["path"]), next=real(item["next"]))
            for item in answer.get("text_encoder_dependents", [])]


forget_preferences()
if os.path.exists(home):
    sys.exit(home + " exists: give an empty work dir")
os.makedirs(os.path.join(home, "Library", "Preferences"))
shutil.copy(binary, app)
# The local API is off until it is turned on (General Settings).
subprocess.run(["defaults", "write", "H3cAppDevCheck", "H3cApp.apiEnabled", "-bool", "true"], check=True)

# The image model's files, as a finished download: clones, and the record
# of a check already made.
if os.stat(weights).st_dev != os.stat(work).st_dev:
    sys.exit("the work dir must be on the volume of " + weights + " (the weights are cloned, not copied)")
for local, (source, size, sha) in WEIGHTS.items():
    os.makedirs(os.path.dirname(os.path.join(image, local)), exist_ok=True)
    subprocess.run(["cp", "-c", os.path.join(weights, source), os.path.join(image, local)], check=True)
    if os.path.getsize(os.path.join(image, local)) != size:
        sys.exit(source + " is not the recorded size")
for local, (source, sha) in SMALL.items():
    path = os.path.join(weights, source)
    if os.path.exists(path) and hashlib.sha256(open(path, "rb").read()).hexdigest() == sha:
        os.makedirs(os.path.dirname(os.path.join(image, local)), exist_ok=True)
        shutil.copy(path, os.path.join(image, local))
json.dump(dict(repo="Comfy-Org/MiniMax-H3-Image", revision=REVISION, revisionDate="2026-11-10",
               declaredLicense="MiniMax-H3-Image Community License Agreement", declaration="written by check.py",
               licenseText="LICENSE", verifiedAt="2026-11-18T00:00:00Z", note="written by check.py",
               files={local: dict(sha256=sha, bytes=size) for local, (_, size, sha) in WEIGHTS.items()}),
          open(os.path.join(image, "H3IMAGE_PROVENANCE.json"), "w"))

# Another volume.
subprocess.run(["hdiutil", "create", "-size", "200m", "-fs", "APFS", "-volname", os.path.basename(volume),
                disk_image], check=True, capture_output=True)
attach()
detach()

start()

# 1. The video model on the image model's volume: cloned, as before.
near = os.path.join(work, "VideoNear")
near_encoder = video_model(near, 5)
status, answer = api("/api/models/add", {"path": near, "name": "Near"})
near_id = answer.get("id")
check(status == 200 and near_id, "a video model on the image model's volume is registered")
state = run_setup()
own = os.path.join(image, "text_encoder")
check(state["phase"] == "done", "the setup finishes (%s)" % state["phase"])
check(state["installed"] and state["text_encoder"] == "own" and real(state["text_encoder_directory"]) == own,
      "same volume: the text encoder is in the image model's folder")
check(os.path.getsize(os.path.join(own, "model.gguf")) == 5 << 20, "it is the video model's file")
check(state["text_encoder_video_model"] is None, "no video model is recorded for a model with its own")
check(any(real(m["path"]) == image for m in api("/api/image-models")[1]["models"]), "the image model is listed")

# 2. A clone takes no space: nothing to give up.
status, answer = api("/api/image-models/share-text-encoder", post=True)
check(status == 409, "a clone is not given up (%s)" % status)
check(os.path.exists(os.path.join(own, "model.gguf")), "and stays")

# 3. A real copy (as an install on another disk made before): given up.
data = open(os.path.join(near_encoder, "model.gguf"), "rb").read()
os.remove(os.path.join(own, "model.gguf"))
with open(os.path.join(own, "model.gguf"), "wb") as f:
    f.write(data)
status, answer = api("/api/image-models/share-text-encoder", post=True)
state = setup_state()
check(status == 200, "a copy that takes space is given up (%s %s)" % (status, answer.get("error", "")))
check(not os.path.exists(own) and not os.path.exists(own + ".removing"), "the copy is gone")
check(os.path.exists(os.path.join(near_encoder, "model.gguf")), "the video model's file is still there")
check(state["text_encoder"] == "shared" and state["text_encoder_directory"] == near_encoder
      and state["text_encoder_video_model"] == near and state["installed"],
      "the video model's is read in place, and recorded")
note = json.load(open(os.path.join(image, "H3IMAGE_PROVENANCE.json")))["note"]
check("There is no text_encoder/ here" in note and near_encoder in note, "the record says so")
check(dependents(near_id) == [{"name": "MiniMax H3 Image", "path": image, "next": None}],
      "deleting the only video model would stop the image model: %s" % dependents(near_id))

# 4. A second video model, on another volume.
attach()
far = os.path.join(volume, "VideoFar")
far_encoder = video_model(far, 6)
status, answer = api("/api/models/add", {"path": far, "name": "Far"})
far_id = answer.get("id")
check(dependents(near_id) == [{"name": "MiniMax H3 Image", "path": image, "next": far}],
      "with another video model, deleting the first moves the image model to it")
check(dependents(far_id) == [], "the other video model has no image model reading it")
status, answer = api("/api/image-models/share-text-encoder", {"video_model": "/no/such/model"})
check(status == 400, "an unregistered video model is refused (%s)" % status)
status, answer = api("/api/image-models/share-text-encoder", {"video_model": far})
state = setup_state()
check(status == 200 and state["text_encoder_directory"] == far_encoder and state["text_encoder_video_model"] == far,
      "another video model's can be chosen")
check(dependents(far_id) == [{"name": "MiniMax H3 Image", "path": image, "next": near}],
      "and is then the one the image model depends on")

# 5. The chosen video model's disk goes and comes back.
detach()
state = setup_state()
check(state["text_encoder"] == "shared" and state["text_encoder_directory"] == near_encoder,
      "its disk away: the other video model's is read")
check(state["text_encoder_video_model"] == far, "the choice is kept")
attach()
check(setup_state()["text_encoder_directory"] == far_encoder, "its disk back: the chosen one is read again")

# 6. No video model with a text encoder.
api("/api/models/select", {"id": far_id})
status, answer = api("/api/models/remove", {"id": near_id})
check(status == 200, "the first video model is taken off the list (%s %s)" % (status, answer.get("error", "")))
detach()
state = setup_state()
check(state["text_encoder"] == "missing" and state["text_encoder_directory"] is None and not state["installed"],
      "no text encoder to read: the model is not usable")
status, answer = api("/api/generate", {"kind": "image", "prompt": "a red apple"})
check(status == 409 and "テキストエンコーダ" in answer.get("error", ""),
      "generating says why (%s %s)" % (status, answer.get("error", "")))
attach()
check(setup_state()["text_encoder"] == "shared" and setup_state()["installed"], "the disk back: usable again")

# 7. A folder with no text_encoder/ of its own is an image model while a
# video model has one.
bare = os.path.join(work, "Bare")
os.makedirs(os.path.join(bare, "tokenizer"))
for name in ("transformer", "vae"):
    os.symlink(os.path.join(image, name), os.path.join(bare, name))
os.symlink(os.path.join(image, "tokenizer", "tokenizer.json"), os.path.join(bare, "tokenizer", "tokenizer.json"))
status, answer = api("/api/image-models/add", {"path": bare})
check(status == 200, "a folder without text_encoder/ is accepted (%s %s)" % (status, answer.get("error", "")))

# 8. A new install whose video model is on another volume: no copy, the
# video model recorded. (The record of the choice is taken away first.)
stop()
subprocess.run(["defaults", "delete", "H3cAppDevCheck", record_key], check=True, capture_output=True)
start()
state = setup_state()
check(state["text_encoder_video_model"] is None and state["text_encoder"] == "shared",
      "before the setup nothing is recorded")
state = run_setup()
check(state["phase"] == "done" and state["installed"], "the setup finishes (%s)" % state["phase"])
check(not os.path.exists(own), "another volume: no copy is made")
check(state["text_encoder"] == "shared" and state["text_encoder_directory"] == far_encoder
      and state["text_encoder_video_model"] == far, "the video model's is read in place, and recorded")

print("%d failed" % len(failures) if failures else "all passed")
sys.exit(1 if failures else 0)
