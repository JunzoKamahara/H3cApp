"""Run one h3-image-mlx stage on a work directory, with the DiT stage patched to
save its intermediate values (initial noise, refined text, latent after each
step) as <work>/trace.safetensors. The reference code itself is not edited:
the patch is applied to its source text in memory.
usage: stage_trace.py <text|dit|vae> <work dir>
With H3_FLOAT_ACTIVATIONS=1 in the environment, int8 projections are computed
from the same int8 weights but without quantizing the activations (the
reference quantizes both, "W8A8"): y = (x rotated by H256) @ (int8 * scale).T
in float32. This separates the reference's activation rounding from the
weights themselves."""
import importlib.util, os, sys
from pathlib import Path
import h3_image_mlx
stage, work = sys.argv[1], Path(sys.argv[2])
if os.environ.get("H3_FLOAT_ACTIVATIONS"):
    import mlx.core as mx
    from h3_image_mlx import core
    def float_linear(x, weight, scale):
        rotated = (x.astype(mx.float32).reshape(-1, 256) @ core.H256).reshape(x.shape)
        return ((rotated @ weight.astype(mx.float32).T) * scale.reshape(-1)).astype(x.dtype)
    core.w8a8 = float_linear
if stage == "dit":
    path = Path(h3_image_mlx.__file__).with_name("dit.py")
    source = path.read_text()
    patches = [
        ("    text = prepare_text(w, text)\n    mx.eval(text, final_mod, *mods)\n",
         "    text = prepare_text(w, text)\n    mx.eval(text, final_mod, *mods)\n"
         "    _TRACE['refined_text'] = text\n"
         "    _TRACE['final_mod'] = final_mod\n    _TRACE['block0_mod'] = mods[0]\n    _TRACE['block49_mod'] = mods[49]\n"),
        ("    sound = mx.array(rng.standard_normal((2 * audio, 32), dtype=np.float32)) if audio else None\n",
         "    sound = mx.array(rng.standard_normal((2 * audio, 32), dtype=np.float32)) if audio else None\n"
         "    _TRACE['noise'] = latent\n    _TRACE['sigmas'] = mx.array(np.asarray(sigmas, dtype=np.float32))\n"
         "    _TRACE['angles'] = angles\n"),
        ("        mx.eval(patches, *([sound] if audio else []))\n",
         "        mx.eval(patches, *([sound] if audio else []))\n"
         "        _TRACE['patches_after_step_%d' % (step + 1)] = patches\n"
         "        _TRACE['head_out_step_%d' % (step + 1)] = x[2 * audio :] @ ww.T + bb\n"),
    ]
    for old, new in patches:
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    spec = importlib.util.spec_from_file_location("h3_image_mlx.dit", path)
    module = importlib.util.module_from_spec(spec)
    module._TRACE = {}
    sys.modules["h3_image_mlx.dit"] = module
    exec(compile(source, str(path), "exec"), module.__dict__)
from h3_image_mlx import _stage
_stage.main(stage, work)
if stage == "dit":
    import mlx.core as mx
    mx.save_safetensors(str(work / "trace.safetensors"), module._TRACE)
