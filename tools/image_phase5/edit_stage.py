"""Run one h3-image-mlx stage of an edit (a request with reference images) on
a work directory, saving the intermediate values the engine is compared with.
The reference code itself is not edited: patches are applied to its source
text in memory.
usage: edit_stage.py <text|references|dit|vae> <work dir>

  text         embedding.safetensors (embedding, tags) as the stage writes it,
               and text_trace.safetensors: the merged Qwen vision rows and the
               first DeepStack level of each reference, and the token count
  references   references.safetensors (the sampled, normalized latent of each
               reference) and references_trace.safetensors: the posterior mean
               and log-variance of each, before normalization
  dit          trace.safetensors as tools/image_phase0/stage_trace.py writes
               it, plus the reference rows after their augmentation
               (ref_rows_<n>, n from 1) and the row count of every segment

H3_FLOAT_ACTIVATIONS=1 as in tools/image_phase0/stage_trace.py."""
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


def patched(name, patches):
    path = Path(h3_image_mlx.__file__).with_name(name + ".py")
    source = path.read_text()
    for old, new in patches:
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    spec = importlib.util.spec_from_file_location("h3_image_mlx." + name, path)
    module = importlib.util.module_from_spec(spec)
    module._TRACE = {}
    sys.modules["h3_image_mlx." + name] = module
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


module = None
if stage == "text":
    module = patched("vision", [
        ("        features, levels, (h, width) = visual_features(w, path)\n",
         "        features, levels, (h, width) = visual_features(w, path)\n"
         "        _TRACE['merged_%d' % i] = features\n"
         "        _TRACE['deepstack0_%d' % i] = levels[0]\n"
         "        _TRACE['grid_%d' % i] = mx.array([h, width])\n"),
        ("    coords = np.concatenate(positions)\n",
         "    coords = np.concatenate(positions)\n"
         "    _TRACE['positions'] = mx.array(coords.astype('int32'))\n"),
    ])
elif stage == "references":
    module = patched("ref_vae", [
        ("    mean, logvar = np.split(np.concatenate(strips, axis=0), 2, axis=-1)\n",
         "    mean, logvar = np.split(np.concatenate(strips, axis=0), 2, axis=-1)\n"
         "    _n = len(_TRACE) // 3 + 1\n"
         "    _TRACE['mean_%d' % _n] = mx.array(mean.transpose(2, 0, 1))\n"
         "    _TRACE['logvar_%d' % _n] = mx.array(logvar.transpose(2, 0, 1))\n"
         "    _TRACE['pixels_%d' % _n] = mx.array(img.shape[:2])\n"),
    ])
elif stage == "dit":
    module = patched("dit", [
        ("    text = prepare_text(w, text)\n    mx.eval(text, final_mod, *mods)\n",
         "    text = prepare_text(w, text)\n    mx.eval(text, final_mod, *mods)\n"
         "    _TRACE['refined_text'] = text\n"
         "    _TRACE['final_mod'] = final_mod\n    _TRACE['block0_mod'] = mods[0]\n    _TRACE['block49_mod'] = mods[49]\n"),
        ("        r = r * 0.999 + mx.array(aug_noise) * 0.001\n",
         "        r = r * 0.999 + mx.array(aug_noise) * 0.001\n"
         "        _TRACE['ref_rows_%d' % len(fixed)] = r\n"),
        ("    sound = mx.array(rng.standard_normal((2 * audio, 32), dtype=np.float32)) if audio else None\n",
         "    sound = mx.array(rng.standard_normal((2 * audio, 32), dtype=np.float32)) if audio else None\n"
         "    _TRACE['noise'] = latent\n    _TRACE['sigmas'] = mx.array(np.asarray(sigmas, dtype=np.float32))\n"
         "    _TRACE['angles'] = angles\n"
         "    _TRACE['segments'] = mx.array(np.asarray(segments, dtype=np.int32))\n"),
        ("        mx.eval(patches, *([sound] if audio else []))\n",
         "        mx.eval(patches, *([sound] if audio else []))\n"
         "        _TRACE['patches_after_step_%d' % (step + 1)] = patches\n"
         "        _TRACE['head_out_step_%d' % (step + 1)] = x[2 * audio :] @ ww.T + bb\n"),
    ])
from h3_image_mlx import _stage
_stage.main(stage, work)
if module is not None and module._TRACE:
    import mlx.core as mx
    name = {"text": "text_trace", "references": "references_trace", "dit": "trace"}[stage]
    mx.save_safetensors(str(work / (name + ".safetensors")), module._TRACE)
