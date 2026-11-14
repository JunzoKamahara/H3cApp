"""Phase-1 check of the still-image side (plans/IMAGE_PLAN.ja.md), driven over
the app's local HTTP API with no model loaded: bring images into a project,
list and show them, hand them to the video form, quit, start again.

usage: roundtrip_check.py <H3cApp binary> <empty work dir> <image.png> <rotated.jpg>

<rotated.jpg> is a landscape picture saved as a JPEG on its side (EXIF
orientation 6).

The app is run as a copy named H3cAppDevCheck with CFFIXED_USER_HOME set to
<work dir>/home, so its Application Support and Movies folders are under the
work dir and its preferences are a domain of its own (H3cAppDevCheck, deleted
before and after) - not the installed app's. Nothing is moved to the Trash:
deleting an image is not exercised here. Port 8420 must be free (quit H3cApp
first)."""
import hashlib, json, os, shutil, signal, struct, subprocess, sys, time, urllib.error, urllib.request

binary, work, sample_png, sample_jpg = (os.path.realpath(path) for path in sys.argv[1:5])
home = os.path.join(work, "home")
support = os.path.join(home, "Library", "Application Support", "H3cApp")
movies = os.path.join(home, "Movies", "H3cApp")
app = os.path.join(work, "H3cAppDevCheck")
base = "http://127.0.0.1:8420"
failures = []
process = None


def forget_preferences():
    # The domain, and the empty file the preferences daemon can leave behind.
    subprocess.run(["defaults", "delete", "H3cAppDevCheck"], capture_output=True)
    import pwd
    leftover = os.path.join(pwd.getpwuid(os.getuid()).pw_dir, "Library", "Preferences", "H3cAppDevCheck.plist")
    if os.path.exists(leftover):
        os.remove(leftover)


def check(condition, what):
    print(("ok   " if condition else "FAIL ") + what)
    if not condition:
        failures.append(what)


def api(path, body=None, raw=False):
    token = open(os.path.join(support, "api-token")).read().strip()
    request = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(),
                                     headers={"Authorization": "Bearer " + token})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            data = response.read()
            return data if raw else (response.status, json.loads(data))
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
    # Killed, as a crash would: nothing is saved or tidied on the way out.
    process.send_signal(signal.SIGKILL)
    process.wait()


def finish(*_):
    if process and process.poll() is None:
        stop()
    time.sleep(1)
    forget_preferences()


import atexit
atexit.register(finish)


def png_size(path):
    with open(path, "rb") as f:
        head = f.read(24)
    return struct.unpack(">II", head[16:24])


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def project():
    return api("/api/project")[1]


def form():
    return api("/api/status")[1]["form"]


forget_preferences()
if os.path.exists(home):
    sys.exit(home + " exists: give an empty work dir")
os.makedirs(os.path.join(home, "Library", "Preferences"))
shutil.copy(binary, app)

# Projects of a version from before images: a named one, an untitled one
# holding a video, and an untitled one holding nothing.
draft = dict(prompt="an old video", creationMethod="text", imageInputMode="firstLastFrame", references=[],
             sizeProfile="square", seconds=5, steps=20, denoiseReuse=2, computeMode="attentionCache",
             speedMode="quality", fastAttention=False, ditLayers=50, seedFixed=False, seed="", loras=[])
def old_project(folder, name, untitled, video):
    os.makedirs(folder)
    stamp = "2026-10-01T00:00:00Z"
    file = dict(version=1, name=name, createdAt=stamp, updatedAt=stamp, batchCount=1, draft=draft)
    if untitled:
        file["untitled"] = True
    json.dump(file, open(os.path.join(folder, "project.json"), "w"))
    if video:
        open(os.path.join(folder, "20261001-000000_seed1.mp4"), "wb").write(b"not really a video")
legacy = os.path.join(support, "Untitled")
old_named = os.path.join(movies, "Old project")
old_project(old_named, "Old project", False, True)
old_project(os.path.join(legacy, "untitled-kept"), "無題のプロジェクト", True, True)
old_project(os.path.join(legacy, "untitled-empty"), "無題のプロジェクト 2", True, False)

print("== first launch")
start()
status = api("/api/status")[1]
check(status["generation_kind"] == "video", "a new window makes videos")
p = project()
check(p["open"] and p["untitled"], "the window has an untitled project")
untitled = p["path"]
check(os.path.dirname(os.path.realpath(untitled)) == os.path.join(support, "UntitledProjects"),
      "the untitled project is in UntitledProjects/, not the folder older versions sweep")
check(json.load(open(os.path.join(untitled, "project.json"))).get("untitled") is None,
      "its project.json does not carry the flag older versions act on")
check(os.path.exists(os.path.join(legacy, "untitled-kept")), "an old untitled project with a video is kept")
check(not os.path.exists(os.path.join(legacy, "untitled-empty")), "an old untitled project with nothing in it is swept")

print("== images into the project")
check(api("/api/kind", {"kind": "image"})[1].get("generation_kind") == "image", "POST /api/kind switches to images")
check(api("/api/kind", {"kind": "audio"})[0] == 400, "an unknown kind is refused")
check(api("/api/generate", {"kind": "image", "prompt": "x"})[0] == 501, "generating an image is refused for now")
check(api("/api/project/import-image", {"path": "/nonexistent.png"})[0] == 400, "a missing file is refused")
code, body = api("/api/project/import-image", {"paths": [sample_png, sample_jpg]})
check(code == 202 and body.get("count") == 2, "two image files are accepted")
check(wait(lambda: not project()["importing_images"] and len(project()["images"]) == 2), "both are listed")
p = project()
images = {image["imported_from"]: image for image in p["images"]}
from_png, from_jpg = images[os.path.basename(sample_png)], images[os.path.basename(sample_jpg)]
check(sha256(from_png["path"]) == sha256(sample_png), "a PNG is kept byte for byte")
record = json.load(open(from_png["path"][:-4] + ".json"))
check(record["sha256"] == sha256(from_png["path"]) and record["assetID"] == from_png["asset_id"],
      "its record holds the file's hash and an id")
check((record["width"], record["height"]) == png_size(sample_png), "and its pixel size")
width, height = png_size(from_jpg["path"])
check(width > height and (from_jpg["width"], from_jpg["height"]) == (width, height),
      "a JPEG stored on its side comes in upright (%dx%d)" % (width, height))
check(not any(name.startswith(".") and name.endswith(".writing") for name in os.listdir(os.path.join(untitled, "images"))),
      "no half-written files are left in images/")
check(p["displayed_image"] is not None, "the last one brought in is in the preview")
check(api("/api/project/show-image", {"image": from_png["file"]})[0] == 200
      and project()["displayed_image"] == from_png["file"], "show-image puts another in the preview")
check(api("/api/result/image", raw=True) == open(from_png["path"], "rb").read(), "GET /api/result/image returns it")
check(api("/api/project/import-image", {"path": os.path.join(untitled, "project.json")})[0] == 202
      and wait(lambda: (project()["message"] or "") != "") and len(project()["images"]) == 2,
      "a file that is not an image is reported and adds nothing")

print("== handing an image to the video form")
check(api("/api/project/use-image", {"image": "nope.png", "as": "first_frame"})[0] == 400, "an unknown image is refused")
check(api("/api/project/use-image", {"image": from_png["file"], "as": "sideways"})[0] == 400, "an unknown use is refused")
code, body = api("/api/project/use-image", {"image": from_png["file"], "as": "first_frame"})
f = form()
check(code == 200 and body["generation_kind"] == "video" and f["creation_method"] == "image"
      and f["image_input_mode"] == "firstLastFrame", "first_frame: the composer makes a video from an image")
copy = os.path.join(untitled, "references", from_png["file"])
check(wait(lambda: os.path.realpath(form()["first_frame"]) == os.path.realpath(copy)),
      "the first frame becomes a copy in references/")
check(sha256(copy) == sha256(from_png["path"]), "the copy holds the same bytes")
sidecar = copy[:-4] + ".json"
check(wait(lambda: os.path.exists(sidecar)) and json.load(open(sidecar))["assetID"] == from_png["asset_id"]
      and json.load(open(sidecar))["image"] == os.path.basename(copy), "its record goes with it, with the same id")
check(not api("/api/status")[1]["is_generating"], "nothing is being generated")
api("/api/project/use-image", {"image": from_jpg["file"], "as": "last_frame"})
check(wait(lambda: os.path.basename(form()["last_frame"] or "") == from_jpg["file"]
           and "/references/" in form()["last_frame"]), "last_frame: the last frame is set (and copied)")
code, body = api("/api/project/use-image", {"image": from_jpg["file"], "as": "reference"})
f = form()
check(code == 200 and f["image_input_mode"] == "referenceImage"
      and [os.path.basename(path) for path in f["references"]] == [from_jpg["file"]], "reference: the image is a reference")
check(len(project()["images"]) == 2, "the copies in references/ are not listed as images")
saved = lambda: json.load(open(os.path.join(untitled, "project.json")))
check(wait(lambda: saved()["draft"].get("firstFrame") == "references/" + from_png["file"]),
      "project.json names the first frame relative to the project")

print("== killed and started again")
api("/api/kind", {"kind": "image"})
check(wait(lambda: saved().get("generationKind") == "image"), "project.json records that images were being made")
stop()
start()
check(os.path.exists(os.path.join(untitled, "images", from_png["file"])),
      "an untitled project holding only images survives the launch sweep")
p = project()
check(p["open"] and os.path.realpath(p["path"]) == os.path.realpath(untitled) and len(p["images"]) == 2,
      "it is reopened, with its images")
check(api("/api/status")[1]["generation_kind"] == "image", "and the composer is back on images")
f = form()
check(f["creation_method"] == "image" and os.path.basename(f["first_frame"] or "") == from_png["file"],
      "the video form kept what was handed to it")

print("== a project from before images")
code, body = api("/api/project/open", {"path": old_named})
check(code == 200 and body.get("name") == "Old project", "it opens")
check(api("/api/status")[1]["generation_kind"] == "video" and form()["prompt"] == "an old video",
      "as a video project, with its form")
p = project()
check(p["images"] == [] and [video["file"] for video in p["videos"]] == ["20261001-000000_seed1.mp4"],
      "with its video and no images")
check(os.path.exists(os.path.join(untitled, "images", from_png["file"])),
      "the untitled project left behind keeps its images")
old_file = json.load(open(os.path.join(old_named, "project.json")))
check("generationKind" not in old_file and "imageDraft" not in old_file and "untitledProject" not in old_file,
      "its project.json gains no image keys while only videos are made")

print("%d failed" % len(failures) if failures else "all passed")
sys.exit(1 if failures else 0)
