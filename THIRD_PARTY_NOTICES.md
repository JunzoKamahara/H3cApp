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

The "SwiftVR（AI動画超解像・研究用）" upscaler is the engine's own C/Metal
implementation (`h3_vsr*.c`, `h3_shaders.metal`) of the SwiftVR one-step
video restoration model: the Wan2.2-based transformer with mask-free
shifted-window attention and the Restoration-aware Autoencoder, transcribed
from the model definitions in [H-oliday/SwiftVR](https://github.com/H-oliday/SwiftVR)
(Apache License 2.0) and the streaming protocol of
[ModelTC/LightX2V](https://github.com/ModelTC/LightX2V) (Apache License 2.0),
whose autoencoder follows [madebyollin/taehv](https://github.com/madebyollin/taehv)
(MIT). No code from those projects ships in the app.

The model weights are **not bundled**. They are published separately by the
SwiftVR authors at https://huggingface.co/H-oliday/SwiftVR under the
"SwiftVR Research License", which permits research and other non-commercial
use only; commercial use needs the authors' written permission. The app only
reads weights the user has downloaded and converted for this Mac (the
"SwiftVRのモデルを準備" sheet, or the `build_vsr_cache` tool), and shows this
notice, with the license, before the download.

