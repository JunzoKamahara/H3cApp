# H3cApp user guide

[English](GUIDE.md) | [日本語](GUIDE.ja.md)

For your first video, start with the [README](README.md) or [illustrated getting-started guide](https://junzokamahara.github.io/H3cApp/en/getting-started.html). This guide explains continuation and joined generation.

## Joined generation: how to use it

One generation makes at most 15 s (362 frames). Longer videos come from
joined generation: short videos are made one after another,
each continuing the one before, and joined into one at their overlaps.

1. **Open a project.** The clips and the joined video are kept in one. With
   none open, pressing generate asks you to create one or pick an existing
   one.
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
- **Enlarge 2x afterwards.** In “Project Videos…”, “More” → “Make a 2x
  Enlarged Copy” adds the video at twice the size to the project
  (`<name>_x2.mp4`), with its sound and its AI-generated record as they
  are. It is the same high-quality resize as the ×2 sizes (not AI
  upscaling) and works on joined videos too. An enlarged copy can't be
  continued (continue the original), and no side may go past 4096 pixels.
- **If you stop partway,** the clips made so far (two or more) are joined.
- **Make the same run again.** With Randomness set to Fixed, the first clip
  uses that seed and each later one the next seed up, so the same settings
  make the same run.
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
