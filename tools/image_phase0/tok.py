import json, sys
from tokenizers import Tokenizer
tok = Tokenizer.from_file(sys.argv[1])
for name, prompt in json.load(open(sys.argv[2])).items():
    ids = tok.encode(prompt, add_special_tokens=False).ids[:4096]
    print(name, len(ids), "tokens:", " ".join(map(str, ids)))
