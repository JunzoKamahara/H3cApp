#!/usr/bin/env python3
"""Checks the engine's environment switches against their table
(src/core/h3_options.c): every h3_option("H3_...") in src/ has a row, every
row is read somewhere, the rows are sorted, and nothing in src/ calls getenv
on an H3_ name directly. Run by `make test`."""
import glob, re, sys

table = re.findall(r'^    \{"(H3_[A-Z0-9_]+)",', open("src/core/h3_options.c").read(), re.M)
read, raw = {}, []
for path in sorted(glob.glob("src/**/*", recursive=True)):
    if not path.endswith((".c", ".m", ".mm", ".h")) or path.endswith("h3_options.c"):
        continue
    text = open(path).read()
    for name in re.findall(r'h3_option\("(H3_[A-Z0-9_]+)"\)', text):
        read.setdefault(name, path)
    raw += [(path, name) for name in re.findall(r'getenv\("(H3_[A-Z0-9_]+)"\)', text)]
problems = []
problems += [f"{path} reads {name} with getenv: use h3_option" for path, name in raw]
problems += [f"{name} is read in {path} but has no row in h3_options.c" for name, path in read.items() if name not in table]
problems += [f"{name} has a row in h3_options.c but nothing reads it" for name in table if name not in read]
if table != sorted(table):
    problems.append("the rows of h3_options.c are not sorted by name")
if len(set(table)) != len(table):
    problems.append("h3_options.c has a name twice")
for problem in problems:
    print("options:", problem)
if problems:
    sys.exit(1)
print(f"ok: {len(table)} environment switches, each documented and read")
