# Phase-1 check of the still-image side

`roundtrip_check.py` drives a development build of H3cApp over its local
HTTP API and checks what phase 1 of `plans/IMAGE_PLAN.ja.md` asks for:
images come into a project, are listed and shown, are handed to the video
form, survive the app being killed, and projects from before images open
as they were. Since phase 5 it also hands images to the image form as the
references of an edit (set, ordered, copied into the project, refused when
malformed, back after a restart). No model is needed, and nothing is
generated.

    make libh3.a
    swift build --package-path native/H3Spike --scratch-path "${TMPDIR}H3cApp-build-scratch" --product H3cApp
    python3 tools/image_phase1/roundtrip_check.py \
        "${TMPDIR}H3cApp-build-scratch/debug/H3cApp" <empty work dir> <image.png> <rotated.jpg>

`<rotated.jpg>` is a landscape picture saved as a JPEG on its side (EXIF
orientation 6); with Pillow:

    im = Image.open("landscape.png").convert("RGB").rotate(90, expand=True)
    exif = Image.Exif(); exif[0x0112] = 6
    im.save("rotated.jpg", exif=exif)

The app under test is a copy named `H3cAppDevCheck`, run with
`CFFIXED_USER_HOME` set to `<work dir>/home`:

- its Application Support and Movies folders are under the work dir, so the
  installed app's projects and models are not touched;
- its preferences are a domain of their own (`H3cAppDevCheck`), removed
  before and after. `CFFIXED_USER_HOME` does not move preferences: a plain
  `swift run H3cApp` reads and writes the domain `H3cApp` in the real home.

Port 8420 must be free, so quit H3cApp first. Deleting an image is not
exercised: the file would go to the Trash of the volume the work dir is on.

Keep the scratch path outside `~/Documents` (see `native/H3Spike/package_app.sh`).
