# Build a sparse stand-in of a remote safetensors file: real header, real bytes
# for the listed data ranges, holes (zeros) elsewhere.
# usage: standin.py <url> <header.json> <ranges.txt> <out> <total_file_bytes>
import os, subprocess, sys
url, header, ranges, out, total = sys.argv[1:6]
raw = open(header, "rb").read()
base = 8 + len(raw)
fd = os.open(out, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
os.ftruncate(fd, int(total))
os.pwrite(fd, len(raw).to_bytes(8, "little") + raw, 0)
done = 0
for line in open(ranges):
    a, b = map(int, line.split())
    for attempt in range(4):
        r = subprocess.run(["curl", "-sSL", "-m", "300", "-A", "generic-client",
                            "-r", "%d-%d" % (base + a, base + b - 1), url],
                           capture_output=True)
        if r.returncode == 0 and len(r.stdout) == b - a: break
    else:
        sys.exit("failed range %d-%d: rc %d, %d bytes" % (a, b, r.returncode, len(r.stdout)))
    os.pwrite(fd, r.stdout, base + a)
    done += b - a
os.close(fd)
print("wrote", done, "bytes of payload into", out)
