"""Write the request.json an h3-image-mlx stage reads.
usage: request.py <prompts.json> <prompt name> <models dir> <width> <height> <seed> <out>"""
import json, sys
prompts, name, models, width, height, seed, out = sys.argv[1:]
json.dump(dict(prompt=json.load(open(prompts))[name], models=models, width=int(width),
               height=int(height), seed=int(seed), references=[], backend="auto",
               dit_mode="stream", precision="int8"), open(out, "w"))
