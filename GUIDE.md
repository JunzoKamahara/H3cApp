# H3cApp user guide

[English](GUIDE.md) | [日本語](GUIDE.ja.md)

For your first video, start with the [README](README.md) or [illustrated getting-started guide](https://junzokamahara.github.io/H3cApp/en/getting-started.html). This guide explains projects, continuation and joined generation.

## Projects

Every video you generate is kept in a project: a folder holding the prompt
and settings, copies of the reference files, and the videos with their
settings records (.json) and latent files (.h3lt, used to continue a video
and for Open-H3 regeneration).

- **Untitled projects.** A new window opens on an “Untitled Project”
  (unless it reopens the last project at launch). Videos made there keep
  their records and latents like any project's, so they can be enlarged or
  continued right away. They are named “Untitled Project”, “Untitled
  Project 2” and so on, with a number no other untitled project has (open
  in a window or kept with its videos). Untitled projects live in the app's own folder
  (`~/Library/Application Support/H3cApp/Untitled`).
- **Saving under a name.** “Project” → “Save as Project…” asks for a
  project name and a location (default `~/Movies/H3cApp`); the untitled
  project's folder moves there with every video made so far.
- **On closing.** An untitled project with no video in it goes away when
  its window closes or another project opens. One with videos stays and is
  listed under “Recent Projects” with the date it was made. After “Close
  Project”, the window goes on with a new untitled project.
- **Moving a video to another project.** In “Project Videos…”, choose
  “More” → “Move to Another Project…” on a video and pick the project (a
  recent one, or a folder). Its settings record and latent file move with
  it, and a joined generation's clips folder too. The reference files the
  record uses (images, audio and so on) are copied into that project. A
  name already taken there gets “-2” and so on. Moving a video exported by
  joining others leaves those videos where they are.

## Joined generation: how to use it

One generation makes at most 15 s (362 frames). Longer videos come from
joined generation: short videos are made one after another,
each continuing the one before, and joined into one at their overlaps.

1. **The project.** The clips and the joined video are kept in the open
   project (a new window's untitled project, too).
2. **Pick the length.** In the length menu under the prompt, pick “15 s
   Joined” or “Custom Clips…” from the “Joined Generation” section.
   - **15 s Joined**: three clips of about 7 s (“Clip length for joined
     generation” in Advanced Settings can make them about 5 s).
   - **Custom Clips**: an editor opens with one clip; “Add a Clip” adds the
     next one below. Each clip gets a length (3-15 s) and what happens in
     it. From the second clip on, turning off “Use the same reference
     images” lets that clip use its own
     reference images, or a first and/or last frame. A clip with a first
     frame doesn't continue the one before: it starts from that image and is
     joined with a cut. With only a last frame it still continues and is
     made to end on that image. Each clip's lip-sync audio can carry on from
     the clip before, change to another file, or be off. Each clip's start
     time and where it is in its audio are listed, so scenes can be laid
     out along a song. The clips can be saved to and read from a JSON file.
   The summary under the panel shows the number of clips, e.g. “3 clips
   joined (15.1 s in all)”. The count menu is off for joined generation (one
   at a time).
3. **Write the common directions.** The prompt box is used for every clip
   (the people, their clothes, the style); each clip gets its own directions
   added after it.
4. **Generate.** The first clip is an ordinary generation; from the second
   on, each one carries over the previous clip's last 22 frames (about
   0.9 s) and their sound. While it runs, the preview shows which clip of
   how many and the whole run's elapsed time, time left and finish time.
5. **Done.** The clips are joined with a crossfade over each overlap into
   one video, shown in the preview. Project Videos lists it as one row,
   “Joined video (5 clips, 30.7 s)”, with its clips folded under it.

Good to know:

- **Directions for each clip.** Given the same prompt, every clip tends to
  start over from the prompt's first scene. Writing what happens in each
  clip in Custom Clips moves the story on (a clip left empty is made from
  the prompt alone). With 15 s Joined, “Edit the clips one by one…” in the
  panel switches to Custom Clips.
- **Continuing a video made elsewhere.** In the menu bar, Project → “Continue a
  Video File…” takes a video not made in H3cApp (from another tool, or one you
  filmed). A sheet then shows the video's size and the size to generate at:
  the one closest to the video's own by default (for a large video, generated
  at half and upscaled), or the current one or a preset. The sheet shows
  the video's last frame with the part that size keeps outlined (what gets
  cropped is darkened); for a video of another aspect ratio, drag the frame
  to choose what to keep (double-click centres it). The video is converted
  to that size and 24 fps (the panel's size becomes it too) and brought into
  the project; how much of its end is carried over (about 0.9, 1.6 or 2.3 s)
  can be chosen too - longer makes importing and generating take longer, and
  the continuation adds less - and that much of its end, picture and sound,
  is turned into the model's own data. Then it continues and joins like a
  video made in H3cApp. Because it is converted, the seam differs slightly
  from the original.
- **Make it longer.** With the joined video shown, “Continue” in the panel
  makes one more clip; “Join and Export” then joins the whole run again.
- **Enlarge afterwards.** In “Project Videos…”, “More” → “Make an Enlarged
  Copy…” adds the video enlarged to the project (`<name>_x2.mp4` at 2x).
  The video shown in the window can be enlarged with Video → “Enlarge the
  Video Shown…” (a project video goes into the project; any other is saved
  where you choose). Any other video file can be enlarged with Video →
  “Enlarge a Video File…”, saved where you choose. Choose the upscaler first: the sheet then
  shows the scales it can make of this video, each with the size it makes.
  Besides those of 1.5x, 2x, 3x and 4x it can do, “Custom” takes a scale
  with up to two decimals and shows the resulting size as you type. The AI
  upscalers also offer “Same size”, which keeps the size and redraws only
  the fine detail. Scales start there for the AI upscalers and at 1.1x for
  Standard and MetalFX, and stop at 8x for those two and at 4x for the AI
  upscalers. Every upscaler stops at 4096 pixels a side, SwiftVR and SeedVR2
  at what fits this Mac's GPU memory, and Open-H3 at 1344×768. The upscalers:
  - Standard: the same high-quality resize as the ×2 sizes, smooth, keeping
    the look as it is.
  - MetalFX: Apple's MetalFX, enlarging along the edges and sharpening a
    little.
  - Real-ESRGAN General (AI upscaling): fills in detail while enlarging, for
    live action and general footage. Since it draws in detail, patterns the
    original doesn't have can appear. On a 24 GB M5 it takes about 0.08 s a
    frame at 800x544 and 0.32 s at 1620x1080. The AI enlarges 4x and
    reaches the chosen size by averaging each 2x2 block (for 2x) or by a
    high-quality resize (other scales).
  - Real-ESRGAN Anime (AI upscaling): crisps up lines and flat colours, for
    anime and illustration-style video, in about half General's time.
  - SwiftVR (AI video super-resolution): an AI model that looks at
    the whole video at once (a one-step restoration model based on Wan2.2)
    enlarges it while drawing texture and detail back in. It works 24 frames
    at a time and takes far longer than Real-ESRGAN. Its weights are not
    bundled: the first time, choose SwiftVR in the sheet and press
    "モデルを準備…" (also in モデル管理 → 拡大モデル), read the license notice,
    and the app downloads the weights (about 20 GB, resumable) from the
    version whose model card declared them under the Apache License 2.0,
    checks each file's SHA-256 against that version's record (the result is
    shown in the sheet and kept in the model folder; see
    THIRD_PARTY_NOTICES.md) and converts them
    for this Mac (about 4.6 GB on an M5, 8.9 GB otherwise). Once converted,
    "元の重みを削除…" in the setup sheet deletes the 20 GB original weights
    (tick "変換が終わったら、元の重みを削除する" to have it done
    automatically). The converted cache stays and keeps working; converting
    again, for example after an app update, needs a new download. It waits for a
    running generation to finish, since it needs the whole GPU
    (plans/UPSCALE_PLAN.md). Its output is brought back to the input's average
    brightness (the model alone comes out slightly brighter).
  - **SwiftVR, restore at 4x then shrink**: the same model restores at 4x and
    the result is brought down to the chosen size with the standard resize.
    It redraws eyes and brows less, with a gentler texture truer to the
    source, at about five times SwiftVR's time. When the 4x intermediate
    does not fit the GPU's memory in one piece (on a 24 GB Mac, inputs beyond
    about 448x256), it is restored as overlapping tiles blended smoothly at
    the seams; when 4x passes 4096 pixels a side or even tiles would not fit,
    it drops to 3x, then 2x, and else stays at the chosen scale. At 4x it is
    the same as SwiftVR. At the same size this is the upscaler truest to the
    source (plans/UPSCALE_PLAN.md appendix A.8).
  - **SeedVR2 (AI super-resolution, detail first; very slow, experimental)**:
    SeedVR2-3B (ByteDance Seed, Apache License 2.0), a one-step diffusion
    restorer, redraws each frame on its own. Fine detail such as eyelashes,
    skin texture and fabric comes out stronger than with SwiftVR, and the
    result can drift further from the source (a face's look may change; it
    is the model's known behaviour on clean inputs). Each frame takes
    seconds: on a 24 GB M5 a 672×384 video enlarged 2x takes about 4 s per
    frame, about 8 minutes for 5 seconds, with about 10 GB of GPU memory;
    960×544 to 1920×1088 takes about 10 s per frame (21 minutes for 5
    seconds) and 13 GB. Macs without the M5's tensor hardware take several
    times longer. Its weights are not bundled: press
    "モデルを準備…" (also in モデル管理 → 拡大モデル), read the license notice,
    and the app downloads the MLX 8-bit export of the weights (about 4.7 GB,
    resumable) from the pinned revision whose model card declares them under
    the Apache License 2.0, checks each file's SHA-256 against that
    revision's record, and keeps the record and the license text in the
    model folder. There is no conversion step. Each output frame's colours
    are matched to its input with the reference's wavelet and L*a*b*
    transfer. It waits for a running generation to finish (plans/UPSCALE_PLAN.md
    appendix A.7).
  - **Open-H3 再生成 (regeneration by H3, up to 1344×768; very slow,
    experimental)**: not an enlargement of the frames but H3 itself
    redrawing the clip on a larger canvas. The latents it was generated
    from (the `.h3lt` beside the video) are widened onto the larger grid by
    Open-H3's latent upscaler, noised back to the strength chosen and
    generated again in just two or three steps with NVIDIA's Open-H3 refiner
    LoRA and LightX2V Turbo v1.2. Texture and pattern come out at the
    quality of H3's own generation, with the clip's prompt and seed carried
    over. At the same size the latents go in as they are. The redraw strength
    has three levels. **Light (σ 0.4)** keeps the people and the composition
    and adds the least detail (the default). **Medium (σ 0.7)** keeps the
    composition, but expressions and faces can change. **Strong (σ 0.909)**
    is the source's recipe: it adds the most detail, can come back as a
    different person or picture at any factor, and takes about 1.3 times as
    long (plans/UPSCALE_PLAN.md appendix A.8). The
    size stays within H3's limit (1344×768 pixels) and the factor is
    rounded into it (a 960×544 video becomes 1344×768 at every factor).
    The factor applies to the size the clip was generated at (its latent
    grid), so a video made at a "×2" size starts from the size before the
    doubling (a 1920×1088 video is redrawn from 960×544, up to 1344×768).
    Only videos this app generated, with their latent file, qualify (not
    imported videos; a joined video through the clips it was joined from).
    The video shown in the preview after generating with no project open
    qualifies too; you choose where to save the result then.
    The time is a generation's: about 12 minutes for 5 seconds at 1344×768
    on an M5, with a generation's memory (about 15 GB). The weights are not
    bundled: "モデルを準備…" shows the license notice and then downloads
    NVIDIA's merged LoRA (FP32, about 12.5 GB) and the latent upscaler,
    checks their SHA-256, extracts the refiner part alone (ranks 0–383) as
    BF16 on this Mac (about 4.2 GB) and deletes the merged file. Turbo v1.2
    is taken from the LoRA folder when the same revision is there. The
    refiner LoRA is under the MiniMax H3 Community License
    (plans/UPSCALE_PLAN.md appendix A.4).

  All of them run inside the app, with no other program. Real-ESRGAN's
  pretrained weights (BSD-3-Clause) are included in the app.

  The sound and the AI-generated record stay as they are, and joined videos
  work too. While it runs, the top of the window shows how far it has got,
  with “Cancel”. No side may go past 4096 pixels.
  An enlarged copy can be continued like any video: the continuation is made
  at the original's size, then enlarged the same way into a copy of its own
  (with the same scale and upscaler), and “Join and Export” joins the
  enlarged copies. An enlarged copy of an enlarged copy can't be continued.
- **If you stop partway,** the clips made so far (two or more) are joined.
- **Make the same run again.** With Randomness set to Fixed, the first clip
  uses that seed and each later one the next seed up, so the same settings
  make the same run.
- **RefMods.** A reference image, video or audio can be saved as a RefMod
  ("Save as RefMod…" on its row): the file is encoded once and kept as a
  small .safetensors in the RefMods folder (Model Manager > RefMod), and
  "Add RefMod…" puts it into any generation without encoding it again or
  showing it to the text encoder, so the prompt needs no <Picture n>.
  The RefMods it makes are Full, keeping everything the ordinary reference
  would, and several pictures of one character can be stacked into one.
  Files made with ComfyUI-MiniMaxH3Mod load too, bundles and compressed
  ones included. Strength
  below 1 blurs the reference rather than weighting it. See
  [REFMOD.md](REFMOD.md).
- **First image, references and lip sync.** Reference images are used by
  every clip, so the people look the same when they come back after leaving
  the frame; in Custom Clips, “Use the same reference images” can be turned
  off for a scene they don't appear in. A first or last frame is used by the first
  clip only. With lip sync, the audio's start moves on by
  itself for each clip, and clips that start after the audio ends make their
  own sound.
- **How long it takes.** On a 24 GB M5 at landscape medium, about 105 s per
  second of finished video (about 50 min for 30 s, about 2.5 h for 90 s).
  15 s joined took 23-25% less time than one 362-frame generation. The
  estimate for the current settings is under “Estimate on this Mac” in
  Advanced Settings.
- **Files.** The clips go in the project's `chains/<name>/` with their
  `.json` records and `.h3lt` latents. Deleting the joined video moves its
  clips to the Trash too.

Through the API, use `seconds` / `chained` / `chain_clip_seconds` (and
`clip_prompts` for directions for each clip) on `POST /api/generate` (see [API](API.md#api)).

For implementation details, see [How joined generation works](TECHNICAL.md#how-joined-generation-works).
