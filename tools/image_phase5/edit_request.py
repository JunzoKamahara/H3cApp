"""Write the request.json of an edit for the h3-image-mlx stages.
usage: edit_request.py <prompt> <models dir> <width> <height> <seed> <out> <reference image>..."""
import json, os, sys
prompt, models, width, height, seed, out = sys.argv[1:7]
json.dump(dict(prompt=prompt, models=models, width=int(width), height=int(height),
               seed=int(seed), references=[os.path.abspath(p) for p in sys.argv[7:]],
               backend="auto", dit_mode="stream", precision="int8"), open(out, "w"))
