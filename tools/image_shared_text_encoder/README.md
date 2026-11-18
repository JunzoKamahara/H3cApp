# Check of the shared text encoder

MiniMax H3 Image uses a video model's text encoder. On the video model's
volume the app clones it into the image model's folder; on another volume it
makes no copy and reads the video model's where it is (`SharedTextEncoder`
in `native/H3Spike/Sources/H3cApp/ImageTextEncoderSharing.swift`, and
`h3_image_load_parts` in the engine).

`check.py` drives a development build of H3cApp over its local HTTP API and
checks:

- a setup beside the video model (a clone), and a clone not being given up;
- a real copy being given up only after its contents are compared with the
  video model's (a copy of another size, and one of the same size whose last
  byte differs, stay), not while a `text_encoder.removing` that no switch
  recorded is in the folder, and not when the copy's folder holds anything
  that was not compared (a folder, a hidden file);
- what deleting a video model would do to the image model;
- choosing another video model's encoder (its files are read through for
  their SHA-256);
- the chosen video model's disk going away: a video model whose files are
  known to have the same SHA-256 stands in; one with the same files that was
  never read through, one of the same size and first MiB with another tail,
  and one that plainly differs are not read until chosen; with none the model
  is not usable;
- a folder without `text_encoder/` being accepted;
- a switch cut short: what the record of the switch names is deleted at launch
  when the video model's encoder can be read, and put back when it is whole
  and nothing else can be read; a copy that is not whole, a file of another
  size than recorded, and a folder no record names are left alone;
- a setup with the video model on another volume (no copy).

Nothing is generated and no weights are loaded.

    make libh3.a
    swift build --package-path native/H3Spike --scratch-path "${TMPDIR}H3cApp-build-scratch" --product H3cApp
    python3 tools/image_shared_text_encoder/check.py \
        "${TMPDIR}H3cApp-build-scratch/debug/H3cApp" <empty work dir> <weights dir>

`<weights dir>` holds MiniMax H3 Image's DiT and VAE as Comfy-Org publishes
them (`diffusion_models/`, `vae/`). They are cloned into the work dir, which
must be on the same APFS volume, with a record of their hashes, so the app
does not read 22 GB to check them. The video models are stand-ins of a few
megabytes; the second is on a 200 MB disk image the script makes, mounts and
unmounts (`/Volumes/H3ShareCheck`).

As in `tools/image_phase1`, the app under test is a copy named
`H3cAppDevCheck` run with `CFFIXED_USER_HOME` set to `<work dir>/home`; its
preferences domain (`H3cAppDevCheck`) is removed before and after, and the
script turns the local API on in it. Port 8420 must be free: quit H3cApp
first. The setup fetches the model's `LICENSE` from Hugging Face.

What the API cannot reach is in `native/H3Spike/Tests/H3cAppTests`: the answer
to the question about a different encoder while another window generates; a
switch cut short with files in `text_encoder/` that may be part of an
encoder; a cancelled reading keeping the setup busy until it has stopped, and
redrawing waiting as generating does; an install or a video model reached
through a link; a copy changed after it was compared; a file put in place of
one whose SHA-256 is known; what is left of a copy when another encoder has
been chosen since; a video model's files staying while an image that reads
them is generated. They use the app's models, so they are skipped unless
`CFFIXED_USER_HOME` names a folder for them:

    CFFIXED_USER_HOME=<empty dir> swift test --package-path native/H3Spike \
        --scratch-path "${TMPDIR}H3cApp-build-scratch" --filter H3cAppTests

To see that the pictures are the same with a shared encoder as with a copy,
generate with the engine's tool, once with the encoder in the folder and
once with `text_encoder=<video model>/FL2VA/text_encoder`, and compare the
two `.ppm` files:

    make build/bin/h3_image_cli
    build/bin/h3_image_cli <image model dir> "<prompt>" out.ppm 1024x1024 1 build/h3_shaders.metal \
        text_encoder=<video model>/FL2VA/text_encoder
