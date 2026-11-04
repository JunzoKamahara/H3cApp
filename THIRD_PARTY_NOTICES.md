# Third-party notices

The rectangular Morton decoder and the dynamic symmetric int8 quantization /
Metal 4 TensorOps scheduling design in `h3_shaders.metal` are adapted from
ccv's Metal FlashAttention `NAMatMulKernel` and `NAInt8MatMulKernel`,
distributed under the following BSD-3-Clause license:

Copyright (c) 2010, Liu Liu
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

- Redistributions of source code must retain the above copyright notice,
  this list of conditions and the following disclaimer.
- Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.
- Neither the name of the authors nor the names of its contributors may be
  used to endorse or promote products derived from this software without
  specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

## ccv (linked into builds made with `CCV_DIR`, including the release .dmg)

The optional fast mode (高速モード) links [liuliu/ccv](https://github.com/liuliu/ccv)'s
`libccv.a` into the app, built from commit
`6a611be1aab6470ae279115ae4eaf7e01bc87135` with the patches in
[tools/ccv_eval](tools/ccv_eval/README.md). ccv itself is under the
BSD-3-Clause license reproduced above (Copyright (c) 2010, Liu Liu). ccv's
`COPYING` file also carries the licenses of the third-party code ccv bundles
(among them Apple's metal-cpp under the Apache License 2.0); builds made
with `CCV_DIR` ship that file unchanged as
`H3cApp.app/Contents/Resources/ccv-COPYING.txt`.

Parts of ccv's Metal FlashAttention kernels are marked
"Copyright (c) 2024 Philip Turner. See MIT LICENSE":

Copyright (c) 2024 Philip Turner

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Real-ESRGAN (weights bundled for 2倍に拡大 / Enlarge 2x)

The AI upscalers "Real-ESRGAN 汎用" and "Real-ESRGAN アニメ" run
[xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)'s pretrained
SRVGGNetCompact models in the app (`RealESRGAN.swift`, on MPSGraph). Their
weights are bundled as `H3cApp.app/Contents/Resources/Upscalers/`, converted
to F16 safetensors by [tools/convert_realesrgan.py](tools/convert_realesrgan.py)
from the release v0.2.5.0 files `realesr-general-x4v3.pth` blended with
`realesr-general-wdn-x4v3.pth` (0.5 each, the reference script's default
denoise strength) and `realesr-animevideov3.pth`. Real-ESRGAN is distributed
under the following BSD-3-Clause license; H3cApp is not endorsed by its
authors:

BSD 3-Clause License

Copyright (c) 2021, Xintao Wang
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

## SwiftVR, LightX2V and TAEHV (the "SwiftVR" upscaler; no weights bundled)

The "SwiftVR（AI動画超解像）" upscaler is the engine's own C/Metal
implementation (`h3_vsr*.c`, `h3_shaders.metal`) of the SwiftVR one-step
video restoration model: the Wan2.2-based transformer with mask-free
shifted-window attention and the Restoration-aware Autoencoder, transcribed
from the model definitions in [H-oliday/SwiftVR](https://github.com/H-oliday/SwiftVR)
(Apache License 2.0) and the streaming protocol of
[ModelTC/LightX2V](https://github.com/ModelTC/LightX2V) (Apache License 2.0),
whose autoencoder follows [madebyollin/taehv](https://github.com/madebyollin/taehv)
(MIT). No code from those projects ships in the app.

The model weights are **not bundled**. They are published separately by the
SwiftVR authors at https://huggingface.co/H-oliday/SwiftVR. The app fetches
them from the last revision of that repository whose model card declared
them under the Apache License 2.0 (revision
`5e89ae51d17b564b40cb14029002b48a83c2ff50`, 2026-09-24), checks the SHA-256
of every file against that revision's object ids, saves the Apache License
2.0 text (the project's LICENSE on GitHub) beside the weights and writes the
record (`H3VSR_PROVENANCE.json`). On 2026-10-10 the repository moved new
versions to the "SwiftVR Research License" (research and other
non-commercial use); its section 14, "Historical Releases", keeps versions
previously released under the Apache License 2.0 under that license, and the
weight files are the same objects at both revisions. The app only reads
weights the user has downloaded and converted for this Mac (the
"SwiftVRのモデルを準備" sheet, `POST /api/upscalers/swiftVR/setup`, or the
`build_vsr_cache` tool), and shows this notice, the declaring model card and
both license texts before the download. Which conditions apply to a use is
the user's judgement; the app records facts and labels nothing "commercial".


## SeedVR2 and mflux (the "SeedVR2" upscaler; no weights bundled)

The "SeedVR2（AI超解像・細部重視）" upscaler is the engine's own C/Metal
implementation (`h3_seedvr2.c`, `h3_shaders.metal`) of SeedVR2-3B, the
one-step diffusion restoration model by ByteDance Seed
([ByteDance-Seed/SeedVR](https://github.com/ByteDance-Seed/SeedVR), Apache
License 2.0; the NaDiT transformer with windowed attention and the causal
3D autoencoder, run one frame at a time), written against the MLX
implementation in [filipstrand/mflux](https://github.com/filipstrand/mflux)
(MIT) and checked stage by stage against it. No code from those projects
ships in the app.

The model weights are **not bundled**. The app fetches
[mlx-community/SeedVR2-3B-mlx-int8](https://huggingface.co/mlx-community/SeedVR2-3B-mlx-int8)
at revision `9dbcb304916d94b4a9769b588ea28439054ce3bb` (2026-06-06): ByteDance
Seed's SeedVR2-3B weights converted to MLX's affine 8-bit format (group 64),
whose model card declares `apache-2.0` and records the chain (ByteDance Seed
→ fp16 safetensors → MLX). The engine reads that export directly and
dequantises it on the GPU at load time; there is no converted cache. Each
file's SHA-256 is checked against that revision's object ids before and
after the download, the upstream repository's Apache License 2.0 text is
saved beside the weights (`LICENSE`, hash-checked) and the record written
(`H3VSR_PROVENANCE.json`). The app shows this notice, the declaring model
card and the license text before the download (the "SeedVR2のモデルを準備"
sheet or `POST /api/upscalers/seedVR2/setup`). Which conditions apply to a
use is the user's judgement; the app records facts and labels nothing
"commercial".

## Open-H3-Regenerate-2K and the LBH latent upscaler (the "Open-H3 再生成" upscaler; no weights bundled)

The "Open-H3 再生成" upscaler is H3's own generation run again from a clip's
latents on a larger canvas (`OpenH3Regenerate.swift`, the engine's
`initial_latent_path` / `video_sigmas` parameters), following the recipe of
NVIDIA's [Open-H3-Regenerate-2K](https://huggingface.co/Efficient-Large-Model/Open-H3-Regenerate-2K)
(code: [NVlabs/Sana, sol-engine branch](https://github.com/NVlabs/Sana/tree/sol-engine/models/Open-H3-Regenerate-2K))
without its LMS Sharp parts. The latent upscaler is the engine's own C/Metal
port (`h3_latent_upscale.c`) of LatentResizer3D from
[LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler)
(MIT), checked against the author's node. No code from those projects
ships in the app.

The weights are **not bundled**. The app fetches, from revision
`5809c33730c18b6b1c827b40974855a4288641b3` (2026-10-28) of that repository,
the merged FP32 LoRA `lora/open_h3_regenerate_2k_lora.safetensors` (rank
576: the Open-H3 refiner, ranks 0–383, with LightX2V FL2VA Turbo v1.2 and
LMS Sharp v1.0 concatenated after it), the latent upscaler
`latent_upsampler/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors`
(the same object as LBH's repository; its `LICENSE` there is the Apache
License 2.0 text), `base_model/LICENSE`, `NOTICE`, `README.md` and the
merge records. Each file's SHA-256 is checked against that revision before
and after the download. The refiner is then **extracted on the user's Mac**
(`h3_lora_extract.c`: lora_A rows and lora_B columns 0–383 kept, BF16) and
the merged file deleted; nothing is redistributed, and the extraction
creates no permission. LightX2V Turbo v1.2 is applied from its own file
(`lightx2v/Minimax-h3-Turbo` at `3ec17a324ced54151364f24f8b5fb6bf7e26414f`,
the revision whose README declared `apache-2.0`), taken from the app's
LoRA folder when a copy is there and downloaded otherwise.

The repository's model card declares `license: other` /
`license_name: minimax-h3-community`, and its README states that the
refiner LoRA, the fast VAE decoder and `base_model/` follow the
[MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/6818f6c32d12b210915e44ad56a4228c2608f160/LICENSE)
(territory, use and redistribution terms included), while the components
NVIDIA did not train (the latent upscaler, the merged Turbo and LMS Sharp
LoRAs) follow their own licenses. The app shows this notice and the links
before the download (the "Open-H3 再生成のモデルを準備" sheet or
`POST /api/upscalers/openH3/setup`) and writes the record
(`H3VSR_PROVENANCE.json`: every hash, the extracted range, the license
statement). Which conditions apply to a use is the user's judgement; the
app records facts and labels nothing "commercial".

Base model notice, as the repository's `NOTICE` states it: "MiniMax H3 is
licensed under the MiniMax H3 Community License Agreement, Copyright © 2026
MiniMax. All Rights Reserved."

## RefMod file format (ComfyUI-MiniMaxH3Mod)

The RefMod files the app reads and writes (`h3_refmod.c`) follow the format
of the ComfyUI node pack ComfyUI-MiniMaxH3Mod by Luisa (luisacaotica),
MIT License, https://github.com/Luisacaotica/ComfyUI-MiniMaxH3Mod: the
`refmod_meta` header metadata, the `latent`/`ref_N` tensors, and the
strength, pooling and refinement arithmetic its nodes define. No code from
that project is included; the format and arithmetic were reimplemented
from its definitions so that files are interchangeable. The ready-made
RefMods published for it (for example on Hugging Face) come under their
own licenses, which the app does not check.
