# H3cApp

[English](README.md) | [日本語](README.ja.md)

**Create AI videos with sound, directly on your Mac.** H3cApp uses [MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) to turn text, images and other media into short videos. After downloading a model, generation takes place on your Mac—no cloud video-generation service, Python, or complicated setup required.

**Powered by MiniMax H3** — an independent app, not an official MiniMax product.

![H3cApp application window](docs/assets/images/app-window-en.jpg)

## What you can do

- **Text to video:** Describe a scene and generate video with synchronized sound.
- **Image to video:** Choose a starting image, an ending image, or both.
- **Reference-based video:** Guide the result using images, videos or audio (optional Ref2VA model required).
- **Lip sync:** Create video matching a voice recording or song.
- **Longer videos:** Continue previous clips and join them automatically.
- **Other features:** Progress previews, project management, and compatible LoRAs.

## Mac requirements

| Item | Requirement |
|---|---|
| Mac | MacBook Air, MacBook Pro, Mac mini, iMac, Mac Studio, or Mac Pro **with Apple Silicon** |
| Chip | Apple M-series (M1 or later); **M5 recommended** for faster processing |
| Memory | **24 GB or more recommended** |
| macOS | macOS 15 (Sequoia) or later |
| Other software | None for the prebuilt app |

**Memory compatibility:** Tested on an M5 Mac with 24 GB of unified memory. **16 GB has not been tested** and is not guaranteed to work. Macs with **less than 16 GB have not been tested either**. Intel Macs are not supported.

M1–M4 Macs can also use the resident or SSD-streaming compute modes, but processing speed varies by hardware and settings.

## Models and disk space

You download a MiniMax H3 model separately from the app.

| Model configuration | Approximate model storage | Best for |
|---|---:|---|
| **GGUF standard (recommended)** | **45 GB** | Balanced quality and storage |
| GGUF compact | 35 GB | Lower disk usage |
| Official BF16 (FL2VA) | 145 GB | Original unquantized model |
| Official BF16 (FL2VA + Ref2VA) | 290 GB | Original model with reference-based generation |

These figures are **model storage sizes, not RAM requirements**. The GGUF configurations support text-to-video and first/last-image generation. Reference images, videos or audio require an additional **Ref2VA model (about 11–21 GB)**.

On supported M5 Macs, an optional speed-up cache uses about **19 GB of extra disk space per model**; using Ref2VA as well adds another cache of about 19 GB for it. Leave additional space for downloads, temporary files and generated videos. Models can be stored on an external SSD.

See [GGUF model details](GGUF.md) for the available model combinations.

## Install and create your first video

1. Download the latest **H3cApp .dmg** from [Releases](https://github.com/JunzoKamahara/H3cApp/releases/latest).
2. Open the .dmg and drag **H3cApp** into **Applications**.
3. Open H3cApp. The setup wizard will help you download a model. **GGUF standard** is a good starting point.
4. Type a prompt, for example, “A quiet beach at sunset, gentle waves and warm light.” Choose the length and size, then click **Generate**.
5. Preview the result and use **Export Video…** to save it.

Generation can take several minutes or longer depending on the Mac and settings. No Xcode or command-line setup is needed for the downloadable app.

For screenshots and step-by-step instructions, visit the [Getting Started guide](https://junzokamahara.github.io/H3cApp/en/getting-started.html).

## Guides and developer documentation

- [Continuing and joining videos](GUIDE.md)
- [Model downloads and GGUF imports](GGUF.md)
- [User guide and advanced settings](https://junzokamahara.github.io/H3cApp/en/guide.html)
- [Technical reference](TECHNICAL.md) · [Local API](API.md) · [Building from source](BUILD.md)
- [FAQ](https://junzokamahara.github.io/H3cApp/en/faq.html) · [Report an issue](https://github.com/JunzoKamahara/H3cApp/issues)

## Publishing videos and licenses

When publishing a generated video, **clearly disclose that it was AI-generated**, as required by MiniMax H3's usage policy. **Export Video…** can add “AI-generated with MiniMax H3” to the video (on by default). You can also use the platform's AI label or mention it in the description. The app includes AI-generation information in MP4 metadata.

H3cApp is under the [MIT License](LICENSE), based on [antirez/h3.c](https://github.com/antirez/h3.c). **The model has separate license terms**; review the [MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE) before use. See [third-party notices](THIRD_PARTY_NOTICES.md) and [technical metadata/license notes](TECHNICAL.md#license).
