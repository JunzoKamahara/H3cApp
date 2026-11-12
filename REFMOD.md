# Making and using RefMods (saved references)

[English](REFMOD.md) | [日本語](REFMOD.ja.md)

A RefMod is a reference image, video or audio file encoded once and saved as a
small file (`.safetensors`). A saved RefMod can go into any generation as a
reference. It is never encoded again, and the text encoder (Qwen) is not shown
it, so the prompt needs no number such as `<Picture 1>`.

The format is that of the RefMods of the ComfyUI node pack
[ComfyUI-MiniMaxH3Mod](https://github.com/Luisacaotica/ComfyUI-MiniMaxH3Mod).
RefMods made in H3cApp load in ComfyUI, and RefMods made in ComfyUI (bundles of
several included) work in H3cApp.

- [What a RefMod is](#what-a-refmod-is)
- [What you need](#what-you-need)
- [Making a RefMod](#making-a-refmod)
- [Using one in a generation](#using-one-in-a-generation)
- [Managing RefMods](#managing-refmods)
- [RefMods made elsewhere](#refmods-made-elsewhere)
- [Projects, continuing and joining](#projects-continuing-and-joining)
- [Limits and cautions](#limits-and-cautions)
- [API](#api)
- [How it works](#how-it-works)
- [License](#license)

## What a RefMod is

An ordinary reference (a file in "Reference images, videos and audio") goes
through two steps in every generation:

1. It is encoded with the video or audio VAE and handed to the DiT (the part
   that draws the video) as a reference.
2. Images and videos are also shown to Qwen, so that `<Picture 1>` or
   `<Video 1>` in the prompt can point at them.

A RefMod is the result of step 1 saved to a file. A generation skips step 1 and
hands the saved contents straight to the DiT. Step 2 does not happen.

| | Ordinary reference | RefMod |
|---|---|---|
| Encoded at every generation | Yes | No (the saved contents are read) |
| Shown to Qwen | Yes (numbered `<Picture n>` and so on) | No (no number) |
| File | The image, video or audio itself | `.safetensors` (a few KB to a few hundred KB) |

As Qwen is not shown a RefMod, it does not suit pointing at "this person" or
"this object" by number in the prompt. It brings the reference's look and feel
into the generation together with the words of the prompt.

## What you need

- **The Ref2VA model.** Making and using RefMods both need Ref2VA, as the
  reference feature does. If it is missing, add it with "Add Ref2VA…" in Model
  Manager.
- To generate with "Fast (int8 cache)", the Ref2VA cache is used, as for
  ordinary references.

## Making a RefMod

1. In the panel, choose "**From images**" and then "**Reference images, videos
   and audio**".
2. Add the image, video or audio to start from with "**Choose Files…**".
3. Press "**Save as RefMod…**" on its row.
4. In the sheet, set the name and press "**Create**".

The RefMod is saved in the RefMod folder
(`~/Library/Application Support/H3cApp/RefMods/`) and listed under "RefMod" in
Model Manager. With "Replace the original reference with the RefMod" on (the
default), the RefMod takes the original file's row in the references, in the
same place. Off, the original row stays and the RefMod is added at the end
(keeping both makes the same reference count twice). A video reference that
also uses its sound keeps its row and gets the RefMod at the end, as a RefMod
holds the picture only.

Making a RefMod is GPU work like a generation, so pressed during one it starts
when that generation finishes. "Stop" in the sheet stops it.

### The sheet

| Field | Default | Meaning |
|---|---|---|
| Name | The file's name | The name in the list, and the file name |
| Resolution cap | 1024 px | An image or video whose short side is larger is shrunk before encoding. Nothing is enlarged (a short side under 320 px is widened to 320 px) |
| Frames to keep | 16 | Video only. Frames picked evenly across the whole video, rounded down to 5, 22, 39, … (16 gives 5). A video of fewer than 5 frames uses its first frame alone |
| Max seconds | 15 | Audio only. Uses up to this many seconds from the start |

### About compressed RefMods

Every RefMod H3cApp makes is Full. It saves the encoded contents as they are,
which match an ordinary reference encoded at the same size.

ComfyUI-MiniMaxH3Mod also has "Compressed", which averages the contents into a
small grid. That has fewer reference tokens, but detail and motion are lost. A
compressed RefMod made from a character design sheet kept only rough colours,
and someone else was drawn. So H3cApp no longer offers to make compressed
RefMods. Compressed RefMods made in ComfyUI or elsewhere still load and work,
and the list shows them as "Compressed".

Examples made on this Mac (M5, 24 GB) from a 768×512 image and 3 seconds of
audio. The times are first runs, loading the VAE included.

| Made | Reference tokens | File | Time |
|---|---|---|---|
| Image, Full | 384 | 74 KB | 6 s |
| Audio, 3 s | 240 | 16 KB | 2 s |

For comparison, the same image as an ordinary reference in a Landscape, Small
generation was 117 tokens. An ordinary reference is shrunk to suit the output
size, while a RefMod is used at the size it was made (by default a short side of
up to 1024 px). A Full RefMod can therefore have more tokens than the same image
as an ordinary reference. To keep it light, make it with a lower resolution
cap.

### Stacking images into one RefMod

Several pictures of the same character or object can go into one RefMod. This is
the "stack" form of ComfyUI-MiniMaxH3Mod, and the published RefMods of people
(20 to 22 photos at 512 px, Full) are made this way.

1. Put two or more images in the references.
2. Press "**Stack Images into a RefMod…**" under the list.
3. In the sheet, tick the images to stack and press "**Create**".

The images are encoded one by one and placed in order along time, one latent
frame each. The list shows it as "4 images", for example. The first image sets
the size: the others are scaled to cover it and cropped at the
centre. The resolution cap defaults to 512 px. Reference tokens grow with each
image. With "Replace the stacked references with the RefMod" on, the chosen
image rows go and the RefMod takes the first one's place. Up to 32 images go in
one stack, and videos can't be stacked.

Four views cut from a character design sheet (front, side, back and face, at
512×864) made a stack of 1,728 tokens and 325 KB in 20 seconds. Generated with
the same prompt and seed, the hair and clothes came out well. But the grid,
labels and side-by-side figures of the crops came along too, and a prompt for a
night-time outdoor stage still gave a sheet-like picture. The same four images
as ordinary references did the same. Stack pictures that show only the
character: no text or frames, and a background close to plain.

## Using one in a generation

1. Choose "**From images**" and then "**Reference images, videos and audio**".
2. Press "**Add RefMod…**" and pick a RefMod from the menu. A file not in the
   list can be picked with "Choose Files…" in the menu ("Choose Files…" next to
   it also takes `.safetensors`).
3. Combine it with ordinary references (images, videos, audio) if you like. The
   order is the list's order.
4. Write the prompt and generate.

A RefMod row shows its name, its contents (kind, size, tokens, Full or
compressed) and a strength slider. "Tokens the references add" and "Memory
needed" under the list include the RefMods.

### Strength

Strength goes from 0.05 to 1; the default is 1.

- **1** uses the saved contents as they are.
- **Below 1**, the contents are mixed with a blurred copy of themselves. The
  lower the value, the more of the reference's detail goes, and the weaker it
  is.

This is the arithmetic of ComfyUI-MiniMaxH3Mod's Apply node. It is an amount of
blur, not a weight such as "reproduce it at so many percent". At 0.3 the
reference's shapes and colours remain, but its detail is quite blurred.

### Bundles

Picking a bundle, a file of several RefMods, opens "Choose the bundle's
members". Tick the members to use and press "Add": each becomes a row of its
own in the references, with its own strength.

### Audio RefMods

RefMods made from audio work too. Unlike an ordinary audio reference, they put
no `<Audio 1>` into the prompt. An ordinary audio reference needs an image or
video reference beside it; an audio RefMod can be used alone.

### Writing the prompt

RefMods get no number such as `<Picture n>`. Numbers go to ordinary references
only, counted from 1 for each kind. For example, in the order "image, RefMod,
image", the two images are `<Picture 1>` and `<Picture 2>`. Describe what the
RefMod holds in words, such as "a red ball…".

## Managing RefMods

Open "**Model Manager**" from the toolbar and choose "**RefMod**" for the list.

- Each row shows the name, where the file is and its contents (member by member
  for a bundle).
- The row's "…" menu: "Rename", "Show in Finder", "Move the file to the Trash
  and remove" ("Remove from the list" for a file outside the RefMod folder).
- "**Open Folder**" opens the RefMod folder in the Finder. A `.safetensors` put
  there is listed the next time this opens.
- A file picked with "**Add…**" is copied into the RefMod folder and listed.
- Drag rows to reorder; the "Add RefMod…" menu follows this order.

## RefMods made elsewhere

Files made with ComfyUI-MiniMaxH3Mod's "Extract H3 RefMod", and RefMods others
publish, work as they are. Bring them in with "Add…" under "RefMod" in Model
Manager, or put them in the RefMod folder.

- Formats read: single files (formats 2 to 4) and bundles (format 5), with
  image, video or audio contents.
- A file that is damaged or has the wrong shape (an image with two frames, for
  example) shows the reason in the list and can't be used in a generation.
- H3cApp makes single files of format 4. It does not make bundles.

Published RefMods come with their own publisher's license and notes; check them
before use. With a RefMod based on a real person, mind how their likeness and
reputation are treated.

## Projects, continuing and joining

- **Projects.** Generating with a project open copies the RefMod files used into
  the project's `references` folder, as with other references. The strength and
  the bundle member are recorded too, and "Restore Settings" brings them back.
  Deleting the original RefMod does not break the project's videos' settings.
- **Continuing a video and joined generation.** RefMods carry over like ordinary
  references.
- **Custom Clips.** A clip with "Use the same reference images" off can take
  RefMods with "Add RefMod…" in its references. As in the panel, each row has
  its own strength, and a bundle asks which members to add. Exporting the clips
  to a file records the strength and the bundle members too.

## Limits and cautions

- RefMods count toward the reference limits (9 images, 3 videos, 3 audio, 12 in
  all) by what they hold. Reference audio, RefMods included, is at most 15
  seconds in total.
- A video RefMod holds only the frames picked when it was made (5 by default).
  It does not remember the motion of a long video as a whole.
- Compressed RefMods made elsewhere have lost detail and motion. They don't
  suit people or characters.
- As Qwen is not shown a RefMod, an image's background, text and layout tend to
  come along with the character. To put the character in another scene, make the
  RefMod from pictures of the character alone, or use ordinary references.
- A RefMod stays as it was saved. After editing the original image, make it
  again.
- RefMods get no `<Picture n>`, so to tell several people apart by number, use
  ordinary references.

## API

The [local API](API.md) can do all of this (it needs `Authorization: Bearer
<token>`). As in [API.md](API.md), the examples pass that header on standard
input: other accounts on the Mac can read a command line with `ps`.

- List: `GET /api/refmods`
- Make: `POST /api/refmods/extract` (`refmod_extracting` in `GET /api/status` is
  true while it runs). Stop it with `POST /api/refmods/extract/cancel`
- Generate: put paths or `{"path", "strength", "member"}` in the
  `reference_paths` of `POST /api/generate`.

```bash
TOKEN_FILE="$HOME/Library/Application Support/H3cApp/api-token"
printf 'Authorization: Bearer %s\n' "$(cat "$TOKEN_FILE")" |
curl -X POST http://127.0.0.1:8420/api/refmods/extract \
  -H @- -H "Content-Type: application/json" \
  -d '{"path": "/Users/me/Pictures/cat.png", "name": "cat"}'
```

```bash
TOKEN_FILE="$HOME/Library/Application Support/H3cApp/api-token"
printf 'Authorization: Bearer %s\n' "$(cat "$TOKEN_FILE")" |
curl -X POST http://127.0.0.1:8420/api/generate \
  -H @- -H "Content-Type: application/json" \
  -d '{"prompt": "a cat walks through a garden",
       "reference_paths": [{"path": "/Users/me/Library/Application Support/H3cApp/RefMods/cat.safetensors", "strength": 0.8}]}'
```

`strength` is above 0 and at most 1, and `member` is a bundle member's number
(from 0). [API.md](API.md) describes every field.

## How it works

A RefMod holds the normalized latent the Ref2VA video VAE (images, videos) or
audio VAE (audio) produces, as it is. ComfyUI's VAEs produce the same values, so
RefMods from either work without conversion. A generation puts this latent where
an ordinary reference's would go and shows Qwen nothing. The strength blur
follows the upstream definition and was checked against PyTorch. See "RefMods: saved reference latents" in the
[technical reference](TECHNICAL.md) for the details.

These ComfyUI-MiniMaxH3Mod features are not in H3cApp: `copies` (stacking the
same RefMod several times), retention presets, curves that change the strength
step by step, the experimental path that turns a RefMod back into an image to
show Qwen, stacking videos, merging or masking several sources into one RefMod,
and making bundles.

## License

The RefMod format and arithmetic follow ComfyUI-MiniMaxH3Mod (MIT License, by
Luisa), but none of its code is included
([THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)). A RefMod made in H3cApp is
encoded by the VAE of the MiniMax H3 model (MiniMax H3 Community License). The
rights to the images, videos and audio it was made from stay with their owners.
Which conditions apply when you share or use a RefMod is your judgement, based
on the model's license and the rights to the source material.
