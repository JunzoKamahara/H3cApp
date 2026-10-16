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

LIB_C := h3.c h3_host.c h3_safetensors.c h3_weights.c h3_text_encoder.c \
	h3_dit_schedule.c h3_dit.c h3_lora.c

LIB_C += h3_video_vae.c h3_video_encoder.c h3_audio_vae.c \
	h3_vision_encoder.c h3_multimodal.c h3_attention_cache.c h3_adaln_curve.c \
	h3_quant.c h3_gguf.c
LIB_M := h3_metal.m h3_gpu.m h3_tokenizer.m h3_av_writer.m h3_av_reader.m
LIB_OBJ := $(LIB_C:.c=.o) $(LIB_M:.m=.o)

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
CCV_EXTRA_OBJ := h3_gpu_ccv_attention.o
# Also a prerequisite of h3_generate_cli, so rebuilding (e.g. re-patching)
# ccv relinks it instead of leaving a stale binary.
CCV_LIB := $(CCV_DIR)/lib/libccv.a
LDLIBS += $(CCV_LIB) -lblas -lc++ \
	-framework CoreML -framework IOSurface -framework QuartzCore
endif

.PHONY: all test parity real-parity clean

all: libh3.a

libh3.a: $(LIB_OBJ) $(CCV_EXTRA_OBJ)
	$(AR) rcs $@ $(LIB_OBJ)

# One-time tool: quantizes QKV/attention-output to int8 and writes the cache
# that H3_ATTENTION_CACHE points the runtime at. See h3_build_attention_cache.c.
build_attention_cache: h3_build_attention_cache.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

# One-time tool: builds the curve-form AdaLN file that H3_ADALN_CURVE points
# the runtime at. See h3_adaln_curve.h.
build_adaln_curve: h3_build_adaln_curve.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

# Same-schedule comparison of curve-form against dense AdaLN (needs the
# released weights; not in `make test`).
h3_adaln_curve_check: tests/adaln_curve_check.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

# Tensor-by-tensor comparison of a quantized re-publication (GGUF) against
# the released weights (needs both; not in `make test`).
h3_weight_compare: tests/weight_compare.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

# Plain-C generation driver (see tools/ccv_eval/h3_generate_cli.c) - a
# Swift-free equivalent of native/H3Spike, built to validate
# H3_ATTENTION_BACKEND=ccv_dense end-to-end without hitting a still-
# unexplained Swift-runtime/ccv interaction documented there and in
# SPEEDUP_ROADMAP.md item 5.
h3_generate_cli: tools/ccv_eval/h3_generate_cli.o $(LIB_OBJ) $(CCV_EXTRA_OBJ) \
		$(CCV_LIB)
	$(CC) -o $@ $(filter-out $(CCV_LIB),$^) $(LDLIBS)

h3_tests: tests/test_h3.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_metal_tests: tests/test_metal.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_renoise_tests: tests/test_renoise.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_bf16_tests: tests/test_bf16.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_tokenizer_tests: tests/test_tokenizer.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_text_tests: tests/test_text_metal.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_audio_gpu_tests: tests/test_audio_gpu.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_quant_tests: tests/test_quant.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_determinism_tests: tests/test_determinism.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

# End-to-end same-seed check (needs the released weights; not in `make test`).
h3_repro_check: tests/repro_check.o $(LIB_OBJ) $(CCV_EXTRA_OBJ) $(CCV_LIB)
	$(CC) -o $@ $(filter-out $(CCV_LIB),$^) $(LDLIBS)

h3_real_audio_vae_test: tests/test_real_audio_vae.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_real_audio_encoder_test: tests/test_real_audio_encoder.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_real_video_encoder_test: tests/test_real_video_encoder.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_real_qwen_vision_test: tests/test_real_qwen_vision.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_real_multimodal_text_test: tests/test_real_multimodal_text.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_real_ref_video_text_test: tests/test_real_ref_video_text.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_real_prompt_test: tests/test_real_prompt.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_real_dit_block_test: tests/test_real_dit_block.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_real_dit_schedule_test: tests/test_real_dit_schedule.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_real_dit_test: tests/test_real_dit.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_semantic_dit_test: tests/test_semantic_dit.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_dit_bench: tests/bench_dit.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_dit_bench_864: tests/bench_dit_864.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

tests/bench_dit_864.o: tests/bench_dit.c
	$(CC) $(CFLAGS) -I. -DH3_BENCH_LATENT_H=30 \
		-DH3_BENCH_LATENT_W=54 -c $< -o $@

h3_real_video_vae_test: tests/test_real_video_vae.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

h3_semantic_vae_test: tests/test_semantic_vae.o $(LIB_OBJ)
	$(CC) -o $@ $^ $(LDLIBS)

test: h3_tests h3_quant_tests h3_metal_tests h3_bf16_tests h3_tokenizer_tests h3_text_tests \
	h3_audio_gpu_tests h3_determinism_tests h3_real_audio_vae_test \
	h3_real_audio_encoder_test h3_real_video_encoder_test \
	h3_real_qwen_vision_test h3_real_multimodal_text_test \
	h3_real_ref_video_text_test

	./h3_tests
	./h3_quant_tests
	@if test -f misc/fixtures/h3_dit.safetensors && \
	         test -f misc/fixtures/h3_dit_bf16.safetensors; then \
		./h3_metal_tests misc/fixtures/h3_dit.safetensors; \
		./h3_bf16_tests misc/fixtures/h3_dit_bf16.safetensors; \
	else \
		echo "skip: MLX toy-block fixtures are not installed"; \
	fi
	@if test -f MiniMax-H3/tokenizer/tokenizer.json; then \
		./h3_tokenizer_tests MiniMax-H3/tokenizer/tokenizer.json; \
	else \
		echo "skip: released tokenizer is not installed"; \
	fi
	@if test -f misc/fixtures/h3_text_bf16.safetensors; then \
		./h3_text_tests misc/fixtures/h3_text_bf16.safetensors; \
	else \
		echo "skip: MLX Qwen fixture is not installed"; \
	fi
	./h3_audio_gpu_tests
	./h3_determinism_tests
	@if test -f MiniMax-H3/FL2VA/audio_vae/model.safetensors && \
	         test -f misc/fixtures/h3_real_audio_vae_37.safetensors; then \
		./h3_real_audio_vae_test; \
	else \
		echo "skip: released AudioVAE weights/fixture are not installed"; \
	fi
	@if test -f MiniMax-H3/FL2VA/audio_vae/model.safetensors && \
	         test -f misc/fixtures/h3_real_audio_encoder_64000.safetensors; then \
		./h3_real_audio_encoder_test; \
	else \
		echo "skip: released audio encoder weights/fixture are not installed"; \
	fi
	@if test -f MiniMax-H3/FL2VA/video_vae/source/model.safetensors && \
	         test -f misc/fixtures/h3_real_video_encoder_256.safetensors; then \
		./h3_real_video_encoder_test; \
	else \
		echo "skip: released visual encoder weights/fixture are not installed"; \
	fi
	@if test -f MiniMax-H3/Ref2VA/video_vae/source/model.safetensors && \
	         test -f misc/fixtures/h3_real_video_encoder_video_22x64.safetensors; then \
		./h3_real_video_encoder_test MiniMax-H3 \
			misc/fixtures/h3_real_video_encoder_video_22x64.safetensors; \
	else \
		echo "skip: released reference-video encoder fixture is not installed"; \
	fi
	@if test -f MiniMax-H3/FL2VA/text_encoder/model-00014-of-00014.safetensors && \
	         test -f misc/fixtures/h3_real_qwen_vision_64.safetensors; then \
		./h3_real_qwen_vision_test; \
	else \
		echo "skip: released Qwen vision weights/fixture are not installed"; \
	fi
	@if test -f MiniMax-H3/Ref2VA/text_encoder/model-00014-of-00014.safetensors && \
	         test -f misc/fixtures/h3_real_qwen_vision_video2x64.safetensors; then \
		./h3_real_qwen_vision_test MiniMax-H3 \
			misc/fixtures/h3_real_qwen_vision_video2x64.safetensors; \
	else \
		echo "skip: released Qwen video-pair fixture is not installed"; \
	fi
	@if test -f MiniMax-H3/FL2VA/text_encoder/model-00001-of-00014.safetensors && \
	         test -f misc/fixtures/h3_real_multimodal_text_64.safetensors; then \
		./h3_real_multimodal_text_test; \
	else \
		echo "skip: released multimodal Qwen weights/fixture are not installed"; \
	fi
	@if test -f MiniMax-H3/Ref2VA/text_encoder/model-00001-of-00014.safetensors && \
	         test -f misc/fixtures/h3_real_ref_video_text_64.safetensors; then \
		./h3_real_ref_video_text_test; \
	else \
		echo "skip: Ref2VA video presentation fixture is not installed"; \
	fi

parity: h3_metal_tests h3_bf16_tests h3_text_tests
	./h3_metal_tests misc/fixtures/h3_dit.safetensors
	./h3_bf16_tests misc/fixtures/h3_dit_bf16.safetensors
	./h3_text_tests misc/fixtures/h3_text_bf16.safetensors

real-parity: h3_real_prompt_test h3_real_dit_block_test
	./h3_real_prompt_test MiniMax-H3 misc/fixtures/h3_real_prompt_bf16.safetensors
	./h3_real_dit_block_test MiniMax-H3 misc/fixtures/h3_real_dit_block0_bf16.safetensors

%.o: %.c
	$(CC) $(CFLAGS) -I. -c $< -o $@

%.o: %.m
	$(CC) $(OBJCFLAGS) -I. -c $< -o $@

h3_gpu_ccv_attention.o: h3_gpu_ccv_attention.mm
	$(CC) -std=c++17 -O3 -fobjc-arc -fblocks $(MINOS_FLAG) -I. -I$(CCV_DIR) -I$(CCV_DIR)/lib \
		-D HAVE_CBLAS -D HAVE_PTHREAD -D HAVE_ACCELERATE_FRAMEWORK \
		-D USE_DISPATCH -D HAVE_MPS -c $< -o $@

tests/%.o: tests/%.c
	$(CC) $(CFLAGS) -I. -c $< -o $@

-include $(wildcard *.d tests/*.d)

clean:
	rm -f h3_tests h3_metal_tests h3_bf16_tests h3_renoise_tests h3_tokenizer_tests \
		h3_text_tests h3_real_prompt_test h3_real_dit_block_test \
		h3_audio_gpu_tests h3_determinism_tests h3_repro_check \
		h3_real_audio_vae_test h3_real_audio_encoder_test \
		h3_real_video_encoder_test h3_real_qwen_vision_test \
		h3_real_multimodal_text_test h3_real_ref_video_text_test \
		h3_real_dit_schedule_test h3_real_dit_test h3_semantic_dit_test \
		h3_real_video_vae_test h3_semantic_vae_test \
	h3_dit_bench h3_dit_bench_864 build_adaln_curve h3_adaln_curve_check h3_quant_tests h3_weight_compare \
	libh3.a *.o *.d tests/*.o tests/*.d
