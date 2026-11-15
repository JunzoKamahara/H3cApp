"""Compare the engine's side of an edit with a run of the reference
implementation (run_edit.sh).

usage: compare_edit.py text <reference embedding.safetensors> <engine embedding.safetensors>
       compare_edit.py reference <reference run dir> <n, from 1> <vae config.json> <engine latent.safetensors>

text       "embedding" and "tags" of the two: the tags must agree row for row;
           the error is given for the vision rows (tag 0) and the language rows
           (tag 1) apart.
reference  the engine's latent of reference n (h3_image_reference_encode: the
           posterior mean, normalized) against the reference run's posterior
           mean (references_trace.safetensors, normalized here with the
           config's latents_mean/std) and against the latent it sampled
           (references.safetensors); and how far that sample is from its mean."""
import json, sys
import mlx.core as mx
import numpy as np


def rel(reference, other):
    r, o = reference.astype(np.float64).ravel(), other.astype(np.float64).ravel()
    return (np.linalg.norm(o - r) / np.linalg.norm(r),
            float(r @ o / (np.linalg.norm(r) * np.linalg.norm(o))))


def text(reference_path, engine_path):
    a, b = mx.load(reference_path), mx.load(engine_path)
    ea, eb = (np.array(v["embedding"].astype(mx.float32)) for v in (a, b))
    ta, tb = (np.array(v["tags"]) for v in (a, b))
    print("reference", ea.shape, "engine", eb.shape)
    if ea.shape != eb.shape:
        sys.exit("row counts differ")
    if not (ta == tb).all():
        sys.exit("tags differ at rows %s" % np.flatnonzero(ta != tb)[:10])
    print("tags agree: %d vision rows, %d language rows" % ((ta == 0).sum(), (ta == 1).sum()))
    for name, mask in (("all rows", np.ones_like(ta, bool)), ("vision rows", ta == 0),
                       ("language rows", ta == 1)):
        error, cosine = rel(ea[mask], eb[mask])
        rows = np.linalg.norm(eb[mask] - ea[mask], axis=-1) / np.linalg.norm(ea[mask], axis=-1)
        print("%-14s relative L2 %.4f  cosine %.6f  per row: median %.4f, max %.4f" %
              (name, error, cosine, np.median(rows), rows.max()))


def reference(run, n, config, engine_path):
    stats = json.load(open(config))
    mean, std = (np.array(stats[k], dtype=np.float32)[:, None, None]
                 for k in ("latents_mean", "latents_std"))
    trace = mx.load(run + "/references_trace.safetensors")
    their_mean = (np.array(trace["mean_%d" % n]) - mean) / std
    spread = np.exp(0.5 * np.clip(np.array(trace["logvar_%d" % n]), -30, 20)) / std
    sample = np.array(mx.load(run + "/references.safetensors")[str(n - 1)])
    engine = np.array(mx.load(engine_path)["latent"])
    print("reference", their_mean.shape, "engine", engine.shape)
    if their_mean.shape != engine.shape:
        sys.exit("grids differ")
    print("posterior spread (std, normalized): mean %.4f, max %.4f; latent rms %.4f" %
          (spread.mean(), spread.max(), np.sqrt((their_mean ** 2).mean())))
    for name, a, b in (("engine vs their mean", their_mean, engine),
                       ("engine vs their sample", sample, engine),
                       ("their sample vs their mean", their_mean, sample)):
        print("%-28s relative L2 %.4f  cosine %.6f" % ((name,) + rel(a, b)))
    # Where the engine and the reference mean differ: tile seams or everywhere.
    error = np.sqrt(((engine - their_mean) ** 2).mean(axis=0))
    print("rms error by latent row band (8 bands):",
          " ".join("%.4f" % band.mean() for band in np.array_split(error, 8, axis=0)))
    print("rms error by latent column band:      ",
          " ".join("%.4f" % band.mean() for band in np.array_split(error, 8, axis=1)))


if sys.argv[1] == "text":
    text(*sys.argv[2:4])
else:
    reference(sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5])
