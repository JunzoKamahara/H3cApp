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


def video_model(folder, megabytes, data=None):
    """A stand-in: FL2VA/text_encoder with one file of that many megabytes."""
    encoder = os.path.join(folder, "FL2VA", "text_encoder")
    os.makedirs(encoder)
    with open(os.path.join(encoder, "model.gguf"), "wb") as f:
        f.write(data if data is not None else os.urandom(megabytes << 20))
    return os.path.join(encoder)


def setup_state():
    return next(item for item in api("/api/image-models")[1]["setups"] if item["model"] == "h3")


def run_setup():
    status, answer = api("/api/image-models/setup", {"confirm_license": True})
    check(status == 200, "the setup starts (%s %s)" % (status, answer.get("error", "")))
    wait(lambda: setup_state()["phase"] == "done" or setup_state()["phase"].startswith("failed"), 120)
    return setup_state()


def share(body=None):
    """POST share-text-encoder, and the phase it ends in (the comparison runs in the background)."""
    status, answer = api("/api/image-models/share-text-encoder", body, post=True)
    if status == 200:
        wait(lambda: not setup_state()["phase"].startswith(("comparing", "reading")), 60)
    return status, answer, setup_state()["phase"]


def real(path):
    # The app spells its home as NSHomeDirectory does (/tmp for /private/tmp).
    return path and os.path.realpath(path)


def dependents(model_id):
    status, answer = api("/api/models/remove", {"id": model_id, "delete_files": True, "dry_run": True})
    return [dict(item, path=real(item["path"]), next=real(item["next"]))
            for item in answer.get("text_encoder_dependents", [])]


def dependent(next=None, same=False, choice=False):
    return [{"name": "MiniMax H3 Image", "path": image, "next": next, "same_encoder": same, "needs_choice": choice}]


def record():
    path = os.path.join(image, "H3IMAGE_TEXT_ENCODER.json")
    return json.load(open(path)) if os.path.exists(path) else None


def write_own(data):
    os.makedirs(own, exist_ok=True)
    with open(os.path.join(own, "model.gguf"), "wb") as f:
        f.write(data)


def leave(data, listed_bytes=None):
    """As a switch cut short leaves the folder: text_encoder.removing, which the
    record names with `listed_bytes` (None: the record names nothing)."""
    os.makedirs(leftover)
    with open(os.path.join(leftover, "model.gguf"), "wb") as f:
        f.write(data)
    if record() is None:
        return
    kept = {key: value for key, value in record().items() if key != "removing"}
    if listed_bytes is not None:
        kept["removing"] = {"model.gguf": listed_bytes}
    json.dump(kept, open(os.path.join(image, "H3IMAGE_TEXT_ENCODER.json"), "w"))


own = os.path.join(image, "text_encoder")
leftover = own + ".removing"

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
near_data = open(os.path.join(near_encoder, "model.gguf"), "rb").read()
status, answer = api("/api/models/add", {"path": near, "name": "Near"})
near_id = answer.get("id")
check(status == 200 and near_id, "a video model on the image model's volume is registered")
state = run_setup()
check(state["phase"] == "done", "the setup finishes (%s)" % state["phase"])
check(state["installed"] and state["text_encoder"] == "own" and real(state["text_encoder_directory"]) == own,
      "same volume: the text encoder is in the image model's folder")
check(os.path.getsize(os.path.join(own, "model.gguf")) == 5 << 20, "it is the video model's file")
check(state["text_encoder_video_model"] is None and record() is None, "no video model is recorded for a model with its own")
check(any(real(m["path"]) == image for m in api("/api/image-models")[1]["models"]), "the image model is listed")

# 2. A clone takes no space: nothing to give up.
status, answer, phase = share()
check(status == 409, "a clone is not given up (%s)" % status)
check(os.path.exists(os.path.join(own, "model.gguf")), "and stays")

# 3. Real copies (as an install on another disk made before).
os.remove(os.path.join(own, "model.gguf"))
write_own(near_data[:4 << 20])
status, answer, phase = share()
check(status == 409, "a copy of another size is not given up (%s)" % status)
os.remove(os.path.join(own, "model.gguf"))
write_own(near_data)
leave(os.urandom(1 << 20))
status, answer, phase = share()
check(status == 409 and "text_encoder.removing" in answer.get("error", ""),
      "a text_encoder.removing no switch recorded stops a switch (%s %s)" % (status, answer.get("error", "")))
check(os.path.getsize(os.path.join(leftover, "model.gguf")) == 1 << 20 and os.path.exists(os.path.join(own, "model.gguf")),
      "and is not deleted")
shutil.rmtree(leftover)
os.remove(os.path.join(own, "model.gguf"))
write_own(near_data[:-1] + bytes([near_data[-1] ^ 1]))
status, answer, phase = share()
check(status == 200 and phase.startswith("failed") and "SHA-256" in phase,
      "a copy of the same size whose contents differ is found out (%s %s)" % (status, phase))
check(os.path.getsize(os.path.join(own, "model.gguf")) == 5 << 20 and not os.path.exists(leftover)
      and setup_state()["text_encoder"] == "own" and record() is None, "and stays, with nothing recorded")
os.remove(os.path.join(own, "model.gguf"))
write_own(near_data)
# Only what was compared goes with the folder: anything else in it stops the switch.
os.makedirs(os.path.join(own, "backup"))
open(os.path.join(own, "backup", "notes.txt"), "w").write("the user's")
status, answer, phase = share()
check(status == 200 and phase.startswith("failed") and "backup" in phase,
      "a copy whose folder holds a folder of the user's is not given up (%s %s)" % (status, phase))
check(os.path.exists(os.path.join(own, "backup", "notes.txt")) and os.path.exists(os.path.join(own, "model.gguf"))
      and not os.path.exists(leftover) and record() is None, "and stays, with what it holds")
shutil.rmtree(os.path.join(own, "backup"))
open(os.path.join(own, ".notes"), "w").write("the user's")
status, answer, phase = share()
check(status == 200 and phase.startswith("failed") and ".notes" in phase,
      "nor one holding a hidden file (%s %s)" % (status, phase))
check(os.path.exists(os.path.join(own, ".notes")) and os.path.exists(os.path.join(own, "model.gguf"))
      and not os.path.exists(leftover) and record() is None, "which stays")
os.remove(os.path.join(own, ".notes"))
status, answer, phase = share()
state = setup_state()
check(status == 200 and phase == "idle", "the same contents: the copy is given up (%s %s)" % (status, phase))
check(not os.path.exists(own) and not os.path.exists(leftover), "the copy is gone")
check(open(os.path.join(near_encoder, "model.gguf"), "rb").read() == near_data, "the video model's file is untouched")
check(state["text_encoder"] == "shared" and state["text_encoder_directory"] == near_encoder
      and state["text_encoder_video_model"] == near and state["installed"],
      "the video model's is read in place, and recorded")
check(record()["files"]["model.gguf"].get("sha256") == hashlib.sha256(near_data).hexdigest()
      and "removing" not in record(), "the record in the model's folder holds the file's SHA-256")
note = json.load(open(os.path.join(image, "H3IMAGE_PROVENANCE.json")))["note"]
check("There is no text_encoder/ here" in note and near_encoder in note, "the record of the weights says so")
check(dependents(near_id) == dependent(), "deleting the only video model would stop the image model: %s" % dependents(near_id))

# 4. A second video model, with a different encoder, on another volume.
attach()
far = os.path.join(volume, "VideoFar")
far_encoder = video_model(far, 6)
far_data = open(os.path.join(far_encoder, "model.gguf"), "rb").read()
status, answer = api("/api/models/add", {"path": far, "name": "Far"})
far_id = answer.get("id")
check(dependents(near_id) == dependent(choice=True),
      "another video model with a different encoder is not moved to silently: %s" % dependents(near_id))
check(dependents(far_id) == [], "the other video model has no image model reading it")
status, answer, phase = share({"video_model": "/no/such/model"})
check(status == 400, "an unregistered video model is refused (%s)" % status)
status, answer, phase = share({"video_model": far})
state = setup_state()
check(status == 200 and phase == "idle" and state["text_encoder_directory"] == far_encoder
      and state["text_encoder_video_model"] == far, "another video model's can be chosen (%s %s)" % (status, phase))
check(record()["files"]["model.gguf"].get("sha256") == hashlib.sha256(far_data).hexdigest(),
      "and is recorded with its file's SHA-256")

# 5. A video model holding the same encoder stands in while the chosen
# one's disk is away - once its files are known to be the same, whole.
twin = os.path.join(work, "VideoTwin")
twin_encoder = video_model(twin, 6, far_data)
status, answer = api("/api/models/add", {"path": twin, "name": "Twin"})
twin_id = answer.get("id")
check(dependents(far_id) == dependent(choice=True),
      "a video model with the same files, not read through yet, is not known to be the same: %s" % dependents(far_id))
detach()
check(setup_state()["text_encoder"] == "needs_choice", "and is not taken on its own (%s)" % setup_state()["text_encoder"])
status, answer, phase = share({"video_model": twin})
check(status == 200 and setup_state()["text_encoder_directory"] == twin_encoder, "it can be chosen (its files are read through)")
attach()
status, answer, phase = share({"video_model": far})
check(setup_state()["text_encoder_directory"] == far_encoder, "and the first chosen back")
check(dependents(far_id) == dependent(next=twin, same=True),
      "now deleting the chosen one moves the image model to the one known to be the same: %s" % dependents(far_id))
detach()
state = setup_state()
check(state["text_encoder"] == "shared" and state["text_encoder_directory"] == twin_encoder,
      "its disk away: the video model known to hold the same encoder is read")
check(state["text_encoder_video_model"] == far, "the choice is kept")
attach()
check(setup_state()["text_encoder_directory"] == far_encoder, "its disk back: the chosen one is read again")

# 6. The same size and the same first MiB, another tail: not the same
# encoder, and not taken until it is chosen.
tail = os.path.join(work, "VideoTail")
tail_encoder = video_model(tail, 6, far_data[:-1] + bytes([far_data[-1] ^ 1]))
status, answer = api("/api/models/add", {"path": tail, "name": "Tail"})
tail_id = answer.get("id")
status, answer, phase = share({"video_model": tail})
check(status == 200 and setup_state()["text_encoder_directory"] == tail_encoder, "the one with another tail is read through and chosen")
status, answer, phase = share({"video_model": far})
check(setup_state()["text_encoder_directory"] == far_encoder, "and the first chosen back")
status, answer = api("/api/models/remove", {"id": twin_id})
check(status == 200, "the twin is taken off the list (%s %s)" % (status, answer.get("error", "")))
check(dependents(far_id) == dependent(choice=True),
      "deleting the chosen one does not move the image model to the one with another tail: %s" % dependents(far_id))
detach()
state = setup_state()
check(state["text_encoder"] == "needs_choice" and state["text_encoder_directory"] is None and not state["installed"],
      "its disk away: the same size and first MiB with another tail is not read (%s)" % state["text_encoder"])
status, answer = api("/api/generate", {"kind": "image", "prompt": "a red apple"})
check(status == 409 and "自動では切り替えません" in answer.get("error", ""),
      "generating says why (%s %s)" % (status, answer.get("error", "")))
status, answer, phase = share({"video_model": near})
state = setup_state()
check(status == 200 and state["text_encoder"] == "shared" and state["text_encoder_directory"] == near_encoder,
      "once chosen, a different one is read")
attach()
status, answer, phase = share({"video_model": far})
check(setup_state()["text_encoder_directory"] == far_encoder, "and the first can be chosen back")
status, answer = api("/api/models/remove", {"id": tail_id})

# 7. No video model with a text encoder.
api("/api/models/select", {"id": far_id})
status, answer = api("/api/models/remove", {"id": near_id})
check(status == 200, "the first video model is taken off the list (%s %s)" % (status, answer.get("error", "")))
detach()
state = setup_state()
check(state["text_encoder"] == "missing" and state["text_encoder_directory"] is None and not state["installed"],
      "no text encoder to read: the model is not usable")
status, answer = api("/api/generate", {"kind": "image", "prompt": "a red apple"})
check(status == 409 and "テキストエンコーダが見つかりません" in answer.get("error", ""),
      "generating says why (%s %s)" % (status, answer.get("error", "")))
attach()
check(setup_state()["text_encoder"] == "shared" and setup_state()["installed"], "the disk back: usable again")

# 8. A folder with no text_encoder/ of its own is an image model while a
# video model has one.
bare = os.path.join(work, "Bare")
os.makedirs(os.path.join(bare, "tokenizer"))
for name in ("transformer", "vae"):
    os.symlink(os.path.join(image, name), os.path.join(bare, name))
os.symlink(os.path.join(image, "tokenizer", "tokenizer.json"), os.path.join(bare, "tokenizer", "tokenizer.json"))
status, answer = api("/api/image-models/add", {"path": bare})
check(status == 200, "a folder without text_encoder/ is accepted (%s %s)" % (status, answer.get("error", "")))

# 9. A switch cut short, the video model's encoder still to be read: the
# rest of the copy - what the record names - goes when the app starts.
stop()
leave(far_data, 6 << 20)
start()
check(wait(lambda: not os.path.exists(leftover), 15), "what a switch cut short left is deleted at launch")
check("removing" not in record() and setup_state()["text_encoder"] == "shared", "and the video model's is read")

# 10. A text_encoder.removing that is not what the record names is never
# deleted, also with an encoder to read.
stop()
leave(os.urandom(1 << 20), 6 << 20)
start()
time.sleep(3)
check(os.path.getsize(os.path.join(leftover, "model.gguf")) == 1 << 20 and setup_state()["text_encoder"] == "shared",
      "a file of another size than the record lists is left alone")
stop()
shutil.rmtree(leftover)
leave(far_data)
start()
time.sleep(3)
check(os.path.getsize(os.path.join(leftover, "model.gguf")) == 6 << 20 and setup_state()["text_encoder"] == "shared",
      "a folder the record names nothing of is left alone")
stop()
shutil.rmtree(leftover)

# 10b. Cut short, the video model gone since, and a file in text_encoder/
# that may be only part of an encoder: the copy left behind is not deleted
# for it (nor put back over it).
os.makedirs(own)
with open(os.path.join(own, "partial.gguf"), "wb") as f:
    f.write(os.urandom(1 << 20))
leave(far_data, 6 << 20)
detach()
start()
time.sleep(3)
check(os.path.getsize(os.path.join(leftover, "model.gguf")) == 6 << 20
      and os.listdir(own) == ["partial.gguf"] and "removing" in record(),
      "files in text_encoder/ do not let the copy left behind be deleted")
stop()
shutil.rmtree(own)
shutil.rmtree(leftover)

# 11. Cut short, and the video model gone since: a copy that is whole is put back.
leave(far_data, 6 << 20)
detach()
start()
check(wait(lambda: os.path.exists(os.path.join(own, "model.gguf")), 15), "with nothing else to read, a whole copy is put back")
check(open(os.path.join(own, "model.gguf"), "rb").read() == far_data and not os.path.exists(leftover)
      and "removing" not in record() and setup_state()["text_encoder"] == "own", "as the model's text encoder")

# 12. Cut short, the video model gone, the copy not whole: left alone.
stop()
shutil.rmtree(own)
leave(far_data[:1 << 20], 6 << 20)
start()
time.sleep(3)
check(os.path.getsize(os.path.join(leftover, "model.gguf")) == 1 << 20 and not os.path.exists(own)
      and setup_state()["text_encoder"] == "missing", "a copy that is not whole is neither put back nor deleted")
attach()

# 13. A new install whose video model is on another volume: no copy, the
# video model recorded. (The record of the choice is taken away first, so
# nothing accounts for the text_encoder.removing of 12 any more: it stays.)
stop()
os.remove(os.path.join(image, "H3IMAGE_TEXT_ENCODER.json"))
start()
state = setup_state()
check(state["text_encoder_video_model"] is None and state["text_encoder"] == "shared",
      "before the setup nothing is recorded")
state = run_setup()
check(state["phase"] == "done" and state["installed"], "the setup finishes (%s)" % state["phase"])
check(not os.path.exists(own), "another volume: no copy is made")
check(os.path.getsize(os.path.join(leftover, "model.gguf")) == 1 << 20,
      "a text_encoder.removing with no record of a switch is left alone, also with an encoder to read")
check(state["text_encoder"] == "shared" and state["text_encoder_directory"] == far_encoder
      and state["text_encoder_video_model"] == far and record()["videoModel"] == far,
      "the video model's is read in place, and recorded in the model's folder")
check(record()["files"]["model.gguf"].get("sha256") == hashlib.sha256(far_data).hexdigest(),
      "with its file's SHA-256")

print("%d failed" % len(failures) if failures else "all passed")
sys.exit(1 if failures else 0)
