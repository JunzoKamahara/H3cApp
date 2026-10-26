# H3cApp local API reference

[English](API.md) | [日本語](API.ja.md)

This is a developer reference for scripting H3cApp locally. The app must be running. See the [README](README.md) for installation.

## API

While `H3cApp.app` is running it serves a plain JSON API on
`http://127.0.0.1:8420` — implemented over raw POSIX sockets (see
[HTTPServer.swift](native/H3Spike/Sources/H3cApp/HTTPServer.swift)), not
Network.framework or any third-party server, and with no separate process
to start or stop. It drives the app's primary `GenerationViewModel`/engine
instance — the one the first window shows (see
[AppModels.swift](native/H3Spike/Sources/H3cApp/AppModels.swift) and
[GenerationViewModel+API.swift](native/H3Spike/Sources/H3cApp/GenerationViewModel+API.swift)).
It keeps answering with every window closed, as long as the app runs;
closing a window closes its project, so open one with
`POST /api/project/open` when needed. There is only ever one job at a time,
shared with the window showing that instance — pressing its generate button
and a `POST /api/generate` compete for the same slot, and whichever loses
gets a clear `409`.

Because the client and server are always on the same Mac, media inputs are
plain filesystem paths, not uploads.

| Method | Path | Does |
|---|---|---|
| `POST` | `/api/generate` | Starts a job. JSON body fully replaces the current draft (see below); `202` once started, `400` on a bad request, `409` if a job is already running. |
| `GET` | `/api/status` | Engine/job state, progress fraction, phase/stage text, error message, and whether a result is ready. While several videos or a joined generation run, `batch` gives `index`/`total`/`chained` and the whole run's `elapsed_seconds` and `estimated_remaining_seconds` (the current video's live estimate plus the later ones, scaled by the pace seen so far); the top-level `estimated_remaining_seconds` stays the current video's. |
| `POST` | `/api/cancel` | Cancels the running job, if any. |
| `GET` | `/api/result/video` | Streams the current result as `video/mp4`; `404` if none, or once the next job's `generate()` call deletes it. |
| `GET` | `/api/models` | Registered H3 model directories (id, name, path, whether active). |
| `GET` | `/api/loras` | Registered LoRA files (id, name, path, strength, recommended steps, whether enabled, step rule `sampler` and `audio_shift` as set in the model manager, and `recommended_sampling` from the file's metadata - `null` while the file is still being read). |
| `GET` | `/api/project` | The open project: name, path, count, and its videos (file, seed, completion time, generation time, `continued_from`, `can_continue`, and for a joined video `joined_clips` / `planned_clips`). |
| `POST` | `/api/project/new` | `{"name", "directory"?}` creates a project from the current form (default directory `~/Movies/H3cApp`) and opens it. |
| `POST` | `/api/project/open` | `{"path"}` opens a project folder; the form takes its saved state. |
| `POST` | `/api/project/close` | Closes it; results are temp files again. |
| `POST` | `/api/project/restore` | `{"video"}` puts that video's request into the form with its seed fixed. |
| `POST` | `/api/project/use-as-reference` | `{"video"}` adds that video to the form as a reference video. |
| `POST` | `/api/project/delete-video` | `{"video"}` moves the video and its record to the Trash. |
| `POST` | `/api/project/join` | `{"video"}` joins the chain of continuations ending at that video into `<name>_joined.mp4` in the project, with a record, (cross-fading each carried-over stretch) and returns `{"path", "clips"}` when done. |
| `POST` | `/api/project/upscale-video` | `{"video", "upscaler"?, "scale"?}` makes an enlarged copy of that project video (`<name>_x2.mp4` at 2x, sound and metadata as they are); `scale` is 1.5, 2 (default), 3 or 4. `upscaler` is `standard` (default: the same high-quality resize as the ×2 sizes), `metalFX` (MetalFX spatial scaling), `realESRGANGeneral` or `realESRGANAnime` (Real-ESRGAN AI upscaling, bundled). Answers at once with `{"video"}`, the name it will have; `400` if a side would pass 4096 pixels, `409` while another is being enlarged. `GET /api/status` (and `GET /api/project`) report `upscaling_video`, `upscale_progress` (0-1) and `message`. |
| `POST` | `/api/upscale-file` | `{"path", "output", "upscaler"?, "scale"?}` writes any video file enlarged to `output` (an `.mp4` path), with the same upscalers and scales. The video is written to a hidden file next to `output` and replaces it only when complete, so an existing `output` stays as it was if enlarging fails or is cancelled; `output` naming the input itself (also through `..`, a symbolic link or a hard link) is `400`. Answers at once; progress and the result as above. |
| `POST` | `/api/upscale/cancel` | Stops the enlargement running now; its unfinished file is removed. `409` if none is running. |
| `POST` | `/api/project/import-video` | `{"path", "size_profile"?, "carry_frames"?, "crop_position"?}` brings a video made elsewhere into the open project to continue it: converted to the size closest to its own (`"size_profile": "current"` keeps the form's, a size name uses that) and 24 fps, with the form set to continue it; `carry_frames` (22, 39 or 56; default 22) is how much of its end is carried over; `crop_position` (`[x, y]`, 0-1 each; default `[0.5, 0.5]`) is where the kept part sits across what a different aspect ratio cuts off (`[0, 0.5]` keeps the left). Answers at once with `{"video"}`, the name it will have; once converted it is listed by `GET /api/project` and usable as `continue_from`. |

`POST /api/generate` body fields, all optional except `prompt`:

| Field | Default | Notes |
|---|---|---|
| `prompt` | — | Required. |
| `size_profile` | `"square"` | A preset `<shape>-<size>` with shape `square`, `landscape` or `portrait` and size `small`, `medium` or `large`, plus `-x2` for the 2x upscaled output (e.g. `landscape-large-x2`: 960x544 upscaled to 1920x1088); or `custom-<W>x<H>` / `custom-<W>x<H>-x2` for a finished size of W x H (multiples of 32, or of 64 with `-x2`, which generates at half that size); the generated size must be at most 600,000 pixels with 256-1344 per side. The older names `smallSquare`, `square`, `landscapeNative`, `landscapeUpscaled`, `portraitNative` and `portraitUpscaled` still work and stand for the matching small/medium presets (see `SizeProfile` in [GenerationModels.swift](native/H3Spike/Sources/H3cApp/GenerationModels.swift)). |
| `seconds` | `5` | 1–15. |
| `steps` | `20`, or the first enabled Turbo LoRA's recommended steps | 3–40. |
| `reuse` | `2` | 1–3. The faster speed modes run at 1 whatever this is. |
| `compute_mode` | the app's own default for this GPU | `attentionCache`, `resident`, or `ssdStreaming`. |
| `speed_mode` | `"quality"` | `quality`, `fast`, or `fastest` (see [Speed modes](TECHNICAL.md#speed-modes)). |
| `dit_layers` | `50` | 35–50. Fewer blocks can break the audio. |
| `fast_attention` | `false` | `true` for [fast mode](TECHNICAL.md#fast-mode-experimental); `400` where it isn't available. |
| `seed` | random | 0–18446744073709551615, as a number or a decimal string (a string keeps a large seed exact for clients whose JSON numbers are doubles). |
| `first_frame_path` / `last_frame_path` | none | FL2VA anchors; cannot combine with `reference_paths`. |
| `seconds` / `chained` / `chain_clip_seconds` | `5` / `false` / `7` | 1-15 s are made in one generation; 16-90 s (or 15 with `"chained": true`) by joined generation, from clips of 5 or 7 s, into one joined video. A joined length needs an open project and `count` 1. |
| `clip_prompts` | none | Joined generation only: an array of strings, one per clip in order, each added after `prompt` for that clip (an empty string or a missing entry uses `prompt` alone). |
| `clip_references` | none | Joined generation with reference images only: an array of true/false, one per clip in order, whether that clip uses the reference images (missing entries do). |
| `clips` | none | Custom Clips: an array of objects, one per clip: `seconds` (3-15; default 7), `prompt` (added after `prompt`), and from the second clip `images` (`same` / `references` / `frames`), `reference_paths`, `first_frame_path` (the clip then starts from it, joined with a cut), `last_frame_path`, `audio` (`carry_on` / `file` / `none`), `audio_path`, `audio_offset`. The first clip uses the request's own images and lip sync. Needs an open project. |
| `continue_from` | none | A video of the open project (a file name from `GET /api/project`) to continue: its last 22 frames and their sound are carried over. Needs the same size; not with first/last frames (references are fine). An enlarged copy is continued at its original's size, and the new clip is then enlarged the same way into a copy that continues it (`upscaling_video` in `GET /api/status` while that runs); not with a chained length. |
| `forced_audio_path` / `forced_audio_offset` | none / `0` | Lip sync: this file's audio (an audio file or a video's soundtrack), from `forced_audio_offset` seconds on, is held as the soundtrack while the picture is generated to match it, and the output carries that audio itself. Works with every creation method. |
| `reference_paths` | `[]` | Ordered Ref2VA references; image/video/audio is auto-detected per path. At least one image or video is required if any audio path is included. |
| `loras` | `[]` | The LoRA stack for this job: names from `GET /api/loras`, or `{"name": ..., "strength": 0.8}` objects (strength multiplies the adapter's trained scale; omitted keeps the entry's saved strength). An object may also set `"sampler"` (`"auto"`, `"euler"`, `"renoise"`) and `"audio_shift"` (0 = auto); like strength, these are saved to the entry. Omitting `loras` means no LoRA, even if some are switched on in the window. |
| `sampler` / `audio_shift` | from the LoRAs | Override the [step rule and audio shift](TECHNICAL.md#step-rule-and-audio-shift) for this one job only (`"auto"`, `"euler"`, `"renoise"`; 0 = auto), e.g. for A/B tests. |
| `lora_name` / `lora_scale` | none | Older single-LoRA form of `loras`; ignored when `loras` is given. |
| `count` | `1` | 1–20 videos in a row with different seeds; above 1 needs an open project. |
| `from_form` | `false` | `true` generates from the form as it is (e.g. after `/api/project/restore`); only `count` may accompany it. |

A value outside its range, or of the wrong type (`true`, `"5"` or `2.5`
where an integer is expected), is a `400` naming the allowed range; nothing
is clamped or rounded silently.

```sh
curl -X POST http://127.0.0.1:8420/api/generate -H "Content-Type: application/json" -d '{
  "prompt": "A red fox walks through fresh snow in a pine forest.",
  "size_profile": "square", "seconds": 5, "steps": 20
}'

curl http://127.0.0.1:8420/api/status
curl http://127.0.0.1:8420/api/result/video -o fox.mp4
```
