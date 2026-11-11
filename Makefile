CC := clang
AR := ar
# The oldest macOS the library supports, matching the app
# (native/H3Spike/Package.swift, Info.plist): MPSGraph's scaled dot-product
# attention (DiT and text encoder) is macOS 15 API, used unguarded like
# BF16 (macOS 14). Without this, clang targets the build machine's macOS:
# @available checks fold to true and the objects claim a newer minimum.
MACOSX_MIN ?= 15.0
MINOS_FLAG := -mmacosx-version-min=$(MACOSX_MIN)
CFLAGS := -std=c11 -O3 -MMD -MP -Wall -Wextra -Wpedantic -Wshadow \
	-Wconversion -Wno-sign-conversion -D_DARWIN_C_SOURCE $(MINOS_FLAG)
OBJCFLAGS := $(CFLAGS) -fobjc-arc
FRAMEWORKS := -framework Foundation -framework Metal \
	-framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph \
	-framework Accelerate -framework AVFoundation -framework CoreMedia \
	-framework CoreVideo -framework CoreAudio -framework CoreGraphics \
	-framework ImageIO
LDLIBS := $(FRAMEWORKS) -licucore -lm $(MINOS_FLAG)

# The generation engine (libh3.a) and the upscale module (libh3upscale.a:
# SwiftVR, SeedVR2, the latent upscaler). The upscale module uses the
# engine, never the other way round.
LIB_C := h3.c h3_host.c h3_safetensors.c h3_weights.c h3_text_encoder.c \
	h3_dit_schedule.c h3_dit.c h3_lora.c h3_hyperflow.c h3_model_family.c \
	h3_video_vae.c h3_video_encoder.c h3_audio_vae.c \
	h3_vision_encoder.c h3_multimodal.c h3_attention_cache.c h3_adaln_curve.c \
	h3_quant.c h3_gguf.c h3_lora_extract.c h3_memory.c h3_plan.c h3_refmod.c
LIB_M := h3_metal.m h3_gpu.m h3_tokenizer.m h3_av_writer.m h3_av_reader.m
UPSCALE_C := h3_vsr_chunk.c h3_vsr_window.c h3_vsr_weights.c h3_vsr_reae.c h3_vsr_wan.c h3_vsr.c \
	h3_latent_upscale.c h3_seedvr2.c h3_seedvr2_vae.c h3_seedvr2_video.c
UPSCALE_M := h3_gpu_vsr.m h3_gpu_sv2.m
# The shader source the programs read is these parts joined, in this order.
SHADER_PARTS := src/gpu/h3_shaders_core.metal src/upscale/h3_shaders_vsr.metal \
	src/upscale/h3_shaders_seedvr2.metal

# Everything the build writes goes under $(BUILD): objects and dependency
# files in obj/ (flat: no two sources share a name), programs in bin/, the
# library at $(BUILD)/libh3.a (native/H3Spike/Package.swift links it there).
BUILD ?= build
OBJ := $(BUILD)/obj
BIN := $(BUILD)/bin
# Sources: the engine in src/ (core, io, model, gpu, upscale), the public
# headers in include/. Includes stay "h3_x.h" everywhere; the directories
# are on the include path.
SRC_DIRS := src/core src/io src/model src/gpu src/upscale
INCLUDES := -Iinclude $(addprefix -I,$(SRC_DIRS))
vpath %.c $(SRC_DIRS) tests tools tools/ccv_eval
vpath %.m $(SRC_DIRS) tools
vpath %.mm src/gpu
CORE_OBJ := $(addprefix $(OBJ)/,$(LIB_C:.c=.o) $(LIB_M:.m=.o))
UPSCALE_OBJ := $(addprefix $(OBJ)/,$(UPSCALE_C:.c=.o) $(UPSCALE_M:.m=.o))
# What the tools and tests link: both, as objects.
LIB_OBJ := $(CORE_OBJ) $(UPSCALE_OBJ)

# Optional, opt-in: set CCV_DIR to a built liuliu/ccv checkout (see
# tools/ccv_eval/README.md) to make the experimental H3_ATTENTION_BACKEND=
# ccv_dense path (h3_gpu_ccv_attention.mm) available. Every build is
# completely unaffected when CCV_DIR is unset (the default).
#
# h3_gpu_ccv_attention.o is deliberately kept OUT of libh3.a: a static
# archive only links in a member that resolves some other object's
# undefined symbol, and nothing references this file's symbols directly
# (it self-registers via a constructor) - archived, it would just be
# silently dropped by any consumer linking libh3.a as `-lh3`. Instead it is
# linked as a loose object file wherever the backend should be available
# (h3_generate_cli below; native/H3Spike/Package.swift does the same for
# Swift, reading this same CCV_DIR variable at `swift build` time), which
# always gets included, and libh3.a still picks it up as an order-only
# prerequisite so `make libh3.a CCV_DIR=...` leaves it built and ready for
# Package.swift to reference without a separate build step.
ifneq ($(strip $(CCV_DIR)),)
CCV_EXTRA_OBJ := $(OBJ)/h3_gpu_ccv_attention.o
# Also a prerequisite of h3_generate_cli, so rebuilding (e.g. re-patching)
# ccv relinks it instead of leaving a stale binary.
CCV_LIB := $(CCV_DIR)/lib/libccv.a
LDLIBS += $(CCV_LIB) -lblas -lc++ \
	-framework CoreML -framework IOSurface -framework QuartzCore
endif

.PHONY: all test parity real-parity clean libh3.a

all: libh3.a

libh3.a: $(BUILD)/libh3.a $(BUILD)/libh3upscale.a $(BUILD)/h3_shaders.metal

$(BUILD)/libh3.a: $(CORE_OBJ) $(CCV_EXTRA_OBJ)
	rm -f $@
	$(AR) rcs $@ $(CORE_OBJ)

$(BUILD)/libh3upscale.a: $(UPSCALE_OBJ)
	rm -f $@
	$(AR) rcs $@ $(UPSCALE_OBJ)

$(BUILD)/h3_shaders.metal: $(SHADER_PARTS) | $(OBJ)
	cat $(SHADER_PARTS) > $@

# One-time tool: quantizes QKV/attention-output to int8 and writes the cache
# that H3_ATTENTION_CACHE points the runtime at. See h3_build_attention_cache.c.
$(BIN)/build_attention_cache: $(OBJ)/h3_build_attention_cache.o $(CORE_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

# One-time tool: converts a downloaded SwiftVR model into the .h3vc cache
# the VSR upscaler runs from. See h3_build_vsr_cache.c and plans/UPSCALE_PLAN.md.
$(BIN)/build_vsr_cache: $(OBJ)/h3_build_vsr_cache.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

# Upscales a clip latent file with the LBH 3D-conv latent upscaler (see
# h3_latent_upscale.h); the Open-H3 regeneration experiment of UPSCALE_PLAN §11.
$(BIN)/h3_latent_upscale: $(OBJ)/latent_upscale_cli.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

# One-time tool: builds the curve-form AdaLN file that H3_ADALN_CURVE points
# the runtime at. See h3_adaln_curve.h.
$(BIN)/build_adaln_curve: $(OBJ)/h3_build_adaln_curve.o $(CORE_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

# Same-schedule comparison of curve-form against dense AdaLN (needs the
# released weights; not in `make test`).
$(BIN)/h3_adaln_curve_check: $(OBJ)/adaln_curve_check.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

# Tensor-by-tensor comparison of a quantized re-publication (GGUF) against
# the released weights (needs both; not in `make test`).
$(BIN)/h3_weight_compare: $(OBJ)/weight_compare.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

# Inspect GGUF files as the importer does, or trim a text encoder to the
# layers the engine reads.
$(BIN)/h3_gguf_tool: $(OBJ)/gguf_tool.o $(CORE_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

# Plain-C generation driver (see tools/ccv_eval/h3_generate_cli.c) - a
# Swift-free equivalent of native/H3Spike, built to validate
# H3_ATTENTION_BACKEND=ccv_dense end-to-end without hitting a still-
# unexplained Swift-runtime/ccv interaction documented there and in
# plans/SPEEDUP_ROADMAP.md item 5.
$(BIN)/h3_generate_cli: $(OBJ)/h3_generate_cli.o $(CORE_OBJ) $(CCV_EXTRA_OBJ) \
		$(CCV_LIB)
	$(CC) -o $@ $(filter-out $(CCV_LIB),$^) $(LDLIBS)

$(BIN)/h3_tests: $(OBJ)/test_h3.o $(CORE_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

# One-step VSR host geometry (chunk protocol, window layout, RoPE): no GPU,
# no weights, always run. See plans/UPSCALE_PLAN.md §9.
$(BIN)/h3_vsr_tests: $(OBJ)/test_vsr.o $(OBJ)/h3_vsr_chunk.o $(OBJ)/h3_vsr_window.o
	$(CC) -o $@ $^ -lm

# The .h3vc weight cache round trip on a synthetic Wan model (GPU, no weights).
$(BIN)/h3_vsr_cache_tests: $(OBJ)/test_vsr_cache.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

# The ReAE autoencoder against a double-precision reference (GPU, no weights).
$(BIN)/h3_vsr_reae_tests: $(OBJ)/test_vsr_reae.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

# The Wan DiT runner against a double-precision reference (GPU, no weights).
$(BIN)/h3_vsr_wan_tests: $(OBJ)/test_vsr_wan.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

# The whole VSR pipeline on a synthetic model (GPU, no weights).
$(BIN)/h3_vsr_session_tests: $(OBJ)/test_vsr_session.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

$(BIN)/h3_vsr_tapconv_tests: $(OBJ)/test_vsr_tapconv.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

# Speed of the TensorOps tap convolution on the SeedVR2 video VAE's shapes (development).
$(BIN)/h3_bench_tapconv: $(OBJ)/bench_tapconv.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

$(BIN)/h3_metal_tests: $(OBJ)/test_metal.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

$(BIN)/h3_renoise_tests: $(OBJ)/test_renoise.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

$(BIN)/h3_bf16_tests: $(OBJ)/test_bf16.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

# Dense SDPA execution modes (issue #2): CPU reference, full vs chunked vs
# explicit, and a probe mode for real DiT shapes ($(BIN)/h3_sdpa_tests --probe).
$(BIN)/h3_sdpa_tests: $(OBJ)/test_sdpa.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

# SeedVR2 (h3_seedvr2.c): the window partition always; the stages against a
# reference dump when H3_SEEDVR2_FIXTURE and the weights are present.
$(BIN)/h3_seedvr2_tests: $(OBJ)/test_seedvr2.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

# Temporal SeedVR2 against the official reference dumps
# (tools/seedvr2_temporal_fixtures.md); skips without them unless
# H3_SEEDVR2_STRICT=1.
$(BIN)/h3_seedvr2_temporal_tests: $(OBJ)/test_seedvr2_temporal.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

# HyperFlow H3 Flash (h3_hyperflow.h): grids and (t, r) rows; with
# H3_HYPERFLOW_DIR also the inspector and the two-time AdaLN schedule
# against a CPU reference (skips without it unless H3_HYPERFLOW_STRICT=1).
$(BIN)/h3_hyperflow_tests: $(OBJ)/test_hyperflow.o $(CORE_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

# HyperFlow on real weights against the ComfyUI reference fixture
# (H3_HYPERFLOW_DIR + H3_HYPERFLOW_FIXTURE; skips without them).
$(BIN)/h3_hyperflow_real_tests: $(OBJ)/test_hyperflow_real.o $(CORE_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

# The LoRA rank extraction (h3_lora_extract.c) on a synthetic file.
$(BIN)/h3_lora_extract_tests: $(OBJ)/test_lora_extract.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

# PNG in, PNG out through the SeedVR2 session (tools/seedvr2_cli.m).
$(BIN)/h3_seedvr2_cli: $(OBJ)/seedvr2_cli.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

$(BIN)/h3_tokenizer_tests: $(OBJ)/test_tokenizer.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

$(BIN)/h3_text_tests: $(OBJ)/test_text_metal.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

$(BIN)/h3_audio_gpu_tests: $(OBJ)/test_audio_gpu.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

$(BIN)/h3_quant_tests: $(OBJ)/test_quant.o $(CORE_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

# GGUF header parsing alone: no Apple frameworks or model weights.
$(BIN)/h3_gguf_header_tests: $(OBJ)/test_gguf.o $(OBJ)/h3_gguf.o $(OBJ)/h3_safetensors.o $(OBJ)/h3_quant.o
	$(CC) -o $@ $^ -lm $(MINOS_FLAG)

$(BIN)/h3_determinism_tests: $(OBJ)/test_determinism.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

# End-to-end same-seed check (needs the released weights; not in `make test`).
$(BIN)/h3_repro_check: $(OBJ)/repro_check.o $(CORE_OBJ) $(CCV_EXTRA_OBJ) $(CCV_LIB)
	$(CC) -o $@ $(filter-out $(CCV_LIB),$^) $(LDLIBS)

$(BIN)/h3_real_audio_vae_test: $(OBJ)/test_real_audio_vae.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

$(BIN)/h3_real_audio_encoder_test: $(OBJ)/test_real_audio_encoder.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

$(BIN)/h3_real_video_encoder_test: $(OBJ)/test_real_video_encoder.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

$(BIN)/h3_video_encoder_chunks_test: $(OBJ)/test_video_encoder_chunks.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

$(BIN)/h3_plan_tests: $(OBJ)/test_plan.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

# The model family and sampler tables (no GPU, no weights).
$(BIN)/h3_model_family_tests: $(OBJ)/test_model_family.o $(CORE_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

$(BIN)/h3_refmod_tests: $(OBJ)/test_refmod.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

$(BIN)/h3_real_refmod_test: $(OBJ)/test_real_refmod.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

$(BIN)/h3_real_qwen_vision_test: $(OBJ)/test_real_qwen_vision.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

$(BIN)/h3_real_multimodal_text_test: $(OBJ)/test_real_multimodal_text.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

$(BIN)/h3_real_ref_video_text_test: $(OBJ)/test_real_ref_video_text.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

$(BIN)/h3_real_prompt_test: $(OBJ)/test_real_prompt.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

$(BIN)/h3_real_dit_block_test: $(OBJ)/test_real_dit_block.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

$(BIN)/h3_real_dit_schedule_test: $(OBJ)/test_real_dit_schedule.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

$(BIN)/h3_real_dit_test: $(OBJ)/test_real_dit.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

$(BIN)/h3_semantic_dit_test: $(OBJ)/test_semantic_dit.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

$(BIN)/h3_dit_bench: $(OBJ)/bench_dit.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

$(BIN)/h3_dit_bench_864: $(OBJ)/bench_dit_864.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

$(OBJ)/bench_dit_864.o: tests/bench_dit.c | $(OBJ)
	$(CC) $(CFLAGS) $(INCLUDES) -DH3_BENCH_LATENT_H=30 \
		-DH3_BENCH_LATENT_W=54 -c $< -o $@

$(BIN)/h3_real_video_vae_test: $(OBJ)/test_real_video_vae.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

# SwiftVR parity against the official implementation (needs the downloaded
# weights and the fixture of tools/dump_swiftvr_fixture.py; self-skipping).
$(BIN)/h3_real_vsr_test: $(OBJ)/test_real_vsr.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

# Stage timing of the SwiftVR engine on the real weights (not in `make test`).
$(BIN)/h3_vsr_bench: $(OBJ)/bench_vsr.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

$(BIN)/h3_int8_bench: $(OBJ)/bench_int8.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

# Runs the SwiftVR engine on raw RGB frames (quality evaluation; not in `make test`).
$(BIN)/h3_vsr_run: $(OBJ)/vsr_run.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

$(BIN)/h3_semantic_vae_test: $(OBJ)/test_semantic_vae.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

test: h3_tests h3_sdpa_tests h3_seedvr2_tests h3_seedvr2_temporal_tests h3_lora_extract_tests h3_vsr_tests h3_vsr_cache_tests h3_vsr_reae_tests h3_vsr_wan_tests h3_vsr_session_tests h3_vsr_tapconv_tests h3_real_vsr_test h3_gguf_header_tests h3_quant_tests h3_metal_tests h3_bf16_tests h3_tokenizer_tests h3_text_tests \
	h3_audio_gpu_tests h3_determinism_tests h3_real_audio_vae_test \
	h3_real_audio_encoder_test h3_real_video_encoder_test \
	h3_video_encoder_chunks_test h3_plan_tests h3_model_family_tests h3_refmod_tests h3_real_refmod_test \
	h3_real_qwen_vision_test h3_real_multimodal_text_test \
	h3_real_ref_video_text_test h3_hyperflow_tests h3_hyperflow_real_tests

	$(BIN)/h3_tests
	$(BIN)/h3_hyperflow_tests
	$(BIN)/h3_hyperflow_real_tests
	$(BIN)/h3_sdpa_tests
	H3_NAX=0 $(BIN)/h3_sdpa_tests
	$(BIN)/h3_seedvr2_tests
	$(BIN)/h3_seedvr2_temporal_tests
	$(BIN)/h3_lora_extract_tests
	$(BIN)/h3_vsr_tests
	$(BIN)/h3_vsr_cache_tests
	$(BIN)/h3_vsr_reae_tests
	$(BIN)/h3_vsr_wan_tests
	$(BIN)/h3_vsr_session_tests
	$(BIN)/h3_vsr_tapconv_tests
	$(BIN)/h3_real_vsr_test
	$(BIN)/h3_gguf_header_tests
	$(BIN)/h3_quant_tests
	@if test -f misc/fixtures/h3_dit.safetensors && \
	         test -f misc/fixtures/h3_dit_bf16.safetensors; then \
		$(BIN)/h3_metal_tests misc/fixtures/h3_dit.safetensors; \
		$(BIN)/h3_bf16_tests misc/fixtures/h3_dit_bf16.safetensors; \
	else \
		echo "skip: MLX toy-block fixtures are not installed"; \
	fi
	@if test -f MiniMax-H3/tokenizer/tokenizer.json; then \
		$(BIN)/h3_tokenizer_tests MiniMax-H3/tokenizer/tokenizer.json; \
	else \
		echo "skip: released tokenizer is not installed"; \
	fi
	@if test -f misc/fixtures/h3_text_bf16.safetensors; then \
		$(BIN)/h3_text_tests misc/fixtures/h3_text_bf16.safetensors; \
	else \
		echo "skip: MLX Qwen fixture is not installed"; \
	fi
	$(BIN)/h3_audio_gpu_tests
# Again with TensorOps off: the shader library must also compile without
# H3_METAL_HAS_TENSOR (pre-M5 Macs, and the fallback after a failed Metal 4 compile).
	H3_NAX=0 $(BIN)/h3_audio_gpu_tests
	$(BIN)/h3_determinism_tests
	@if test -f MiniMax-H3/FL2VA/audio_vae/model.safetensors && \
	         test -f misc/fixtures/h3_real_audio_vae_37.safetensors; then \
		$(BIN)/h3_real_audio_vae_test; \
	else \
		echo "skip: released AudioVAE weights/fixture are not installed"; \
	fi
	@if test -f MiniMax-H3/FL2VA/audio_vae/model.safetensors && \
	         test -f misc/fixtures/h3_real_audio_encoder_64000.safetensors; then \
		$(BIN)/h3_real_audio_encoder_test; \
	else \
		echo "skip: released audio encoder weights/fixture are not installed"; \
	fi
	@if test -f MiniMax-H3/FL2VA/video_vae/source/model.safetensors && \
	         test -f misc/fixtures/h3_real_video_encoder_256.safetensors; then \
		$(BIN)/h3_real_video_encoder_test; \
	else \
		echo "skip: released visual encoder weights/fixture are not installed"; \
	fi
	@if test -f MiniMax-H3/Ref2VA/video_vae/source/model.safetensors && \
	         test -f misc/fixtures/h3_real_video_encoder_video_22x64.safetensors; then \
		$(BIN)/h3_real_video_encoder_test MiniMax-H3 \
			misc/fixtures/h3_real_video_encoder_video_22x64.safetensors; \
	else \
		echo "skip: released reference-video encoder fixture is not installed"; \
	fi
	$(BIN)/h3_video_encoder_chunks_test
	$(BIN)/h3_plan_tests
	$(BIN)/h3_model_family_tests
	$(BIN)/h3_refmod_tests
	@if test -f MiniMax-H3/Ref2VA/video_vae/source/model.safetensors; then \
		$(BIN)/h3_real_refmod_test; \
	else \
		echo "skip: released Ref2VA VAE weights are not installed (RefMod extraction)"; \
	fi
	@if test -f MiniMax-H3/Ref2VA/video_vae/source/model.safetensors && \
	         test -n "$(H3_CHUNKS_VIDEO)" && test -f "$(H3_CHUNKS_VIDEO)"; then \
		$(BIN)/h3_video_encoder_chunks_test MiniMax-H3/Ref2VA/video_vae/source \
			"$(H3_CHUNKS_VIDEO)"; \
	else \
		echo "skip: chunked encoder check needs the weights and H3_CHUNKS_VIDEO=<video>"; \
	fi
	@if test -f MiniMax-H3/FL2VA/text_encoder/model-00014-of-00014.safetensors && \
	         test -f misc/fixtures/h3_real_qwen_vision_64.safetensors; then \
		$(BIN)/h3_real_qwen_vision_test; \
	else \
		echo "skip: released Qwen vision weights/fixture are not installed"; \
	fi
	@if test -f MiniMax-H3/Ref2VA/text_encoder/model-00014-of-00014.safetensors && \
	         test -f misc/fixtures/h3_real_qwen_vision_video2x64.safetensors; then \
		$(BIN)/h3_real_qwen_vision_test MiniMax-H3 \
			misc/fixtures/h3_real_qwen_vision_video2x64.safetensors; \
	else \
		echo "skip: released Qwen video-pair fixture is not installed"; \
	fi
	@if test -f MiniMax-H3/FL2VA/text_encoder/model-00001-of-00014.safetensors && \
	         test -f misc/fixtures/h3_real_multimodal_text_64.safetensors; then \
		$(BIN)/h3_real_multimodal_text_test; \
	else \
		echo "skip: released multimodal Qwen weights/fixture are not installed"; \
	fi
	@if test -f MiniMax-H3/Ref2VA/text_encoder/model-00001-of-00014.safetensors && \
	         test -f misc/fixtures/h3_real_ref_video_text_64.safetensors; then \
		$(BIN)/h3_real_ref_video_text_test; \
	else \
		echo "skip: Ref2VA video presentation fixture is not installed"; \
	fi

parity: h3_metal_tests h3_bf16_tests h3_text_tests
	$(BIN)/h3_metal_tests misc/fixtures/h3_dit.safetensors
	$(BIN)/h3_bf16_tests misc/fixtures/h3_dit_bf16.safetensors
	$(BIN)/h3_text_tests misc/fixtures/h3_text_bf16.safetensors

real-parity: h3_real_prompt_test h3_real_dit_block_test
	$(BIN)/h3_real_prompt_test MiniMax-H3 misc/fixtures/h3_real_prompt_bf16.safetensors
	$(BIN)/h3_real_dit_block_test MiniMax-H3 misc/fixtures/h3_real_dit_block0_bf16.safetensors

$(OBJ) $(BIN):
	mkdir -p $@

$(OBJ)/%.o: %.c | $(OBJ)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

$(OBJ)/%.o: %.m | $(OBJ)
	$(CC) $(OBJCFLAGS) $(INCLUDES) -c $< -o $@

$(OBJ)/h3_gpu_ccv_attention.o: h3_gpu_ccv_attention.mm | $(OBJ)
	$(CC) -std=c++17 -O3 -fobjc-arc -fblocks $(MINOS_FLAG) $(INCLUDES) -I$(CCV_DIR) -I$(CCV_DIR)/lib \
		-D HAVE_CBLAS -D HAVE_PTHREAD -D HAVE_ACCELERATE_FRAMEWORK \
		-D USE_DISPATCH -D HAVE_MPS -c $< -o $@

# Every program can be asked for by its bare name (make h3_tests).
PROGRAMS := h3_model_family_tests build_attention_cache build_vsr_cache h3_latent_upscale build_adaln_curve \
	h3_adaln_curve_check h3_weight_compare h3_gguf_tool h3_generate_cli \
	h3_tests h3_vsr_tests h3_vsr_cache_tests h3_vsr_reae_tests \
	h3_vsr_wan_tests h3_vsr_session_tests h3_vsr_tapconv_tests h3_bench_tapconv \
	h3_metal_tests h3_renoise_tests h3_bf16_tests h3_sdpa_tests \
	h3_seedvr2_tests h3_seedvr2_temporal_tests h3_hyperflow_tests h3_hyperflow_real_tests \
	h3_lora_extract_tests h3_seedvr2_cli h3_tokenizer_tests h3_text_tests \
	h3_audio_gpu_tests h3_quant_tests h3_gguf_header_tests h3_determinism_tests \
	h3_repro_check h3_real_audio_vae_test h3_real_audio_encoder_test h3_real_video_encoder_test \
	h3_video_encoder_chunks_test h3_plan_tests h3_refmod_tests h3_real_refmod_test \
	h3_real_qwen_vision_test h3_real_multimodal_text_test h3_real_ref_video_text_test h3_real_prompt_test \
	h3_real_dit_block_test h3_real_dit_schedule_test h3_real_dit_test h3_semantic_dit_test \
	h3_dit_bench h3_dit_bench_864 h3_real_video_vae_test h3_real_vsr_test \
	h3_vsr_bench h3_int8_bench h3_vsr_run h3_semantic_vae_test
.PHONY: $(PROGRAMS)
$(PROGRAMS): %: $(BIN)/%
$(addprefix $(BIN)/,$(PROGRAMS)): | $(BIN) $(BIN)/h3_shaders.metal

# The programs name the shader source "h3_shaders.metal"; it is read from
# the working directory or, failing that, from beside the executable
# (h3_gpu.m), so a copy sits in $(BIN).
$(BIN)/h3_shaders.metal: $(BUILD)/h3_shaders.metal | $(BIN)
	cp $< $@

-include $(wildcard $(OBJ)/*.d)

clean:
	rm -rf $(BUILD)
