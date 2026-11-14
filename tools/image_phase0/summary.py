"""Table of the phase-0 runs: tokens, DiT/VAE time, peak memory, latent statistics.
usage: summary.py <phase0 dir> <prompt name>... """
import json, re, sys
from pathlib import Path
import mlx.core as mx
root = Path(sys.argv[1])
def timing(p):
    t = p.read_text() if p.exists() else ""
    real = re.search(r"([\d.]+) real", t); peak = re.search(r"(\d+)\s+peak memory footprint", t)
    return (float(real.group(1)) if real else float("nan"), int(peak.group(1)) / 1e9 if peak else float("nan"))
print("| prompt | size | tokens | DiT s (per step) | DiT peak GB | VAE s | VAE peak GB | latent mean / std | step-6 head out std |")
print("| - | - | - | - | - | - | - | - | - |")
for name in sys.argv[2:]:
    for work in sorted((root / name).iterdir(), key=lambda p: (len(p.name), p.name)):
        if not (work / "latent.safetensors").exists(): continue
        prof = json.loads((work / "profile.dit.json").read_text())
        dt, dm = timing(work / "time.dit.txt"); vt, vm = timing(work / "time.vae.txt")
        z = mx.load(str(work / "latent.safetensors"))["latent"]
        tr = mx.load(str(work / "trace.safetensors"))
        print("| %s | %s | %d | %.1f (%.1f) | %.2f | %.1f | %.2f | %.3f / %.3f | %.3f |" % (
            name, work.name, prof["tokens"], prof["total_s"], sum(prof["steps"]) / len(prof["steps"]), dm, vt, vm,
            z.mean().item(), z.std().item(), tr["head_out_step_6"].astype(mx.float32).std().item()))
print("trace keys:", sorted(tr.keys()))
print({k: (str(v.dtype), tuple(v.shape)) for k, v in tr.items()})
