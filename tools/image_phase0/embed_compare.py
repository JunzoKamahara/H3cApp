"""Compare two text embeddings of the same prompt, each "embedding" [tokens, 5120].
usage: embed_compare.py <reference.safetensors> <other.safetensors>
Prints the relative L2 error against the reference, the cosine similarity of
the whole matrix and of each token row, the row-norm ratio, and how the
error is spread over the tokens."""
import sys
import mlx.core as mx
import numpy as np

def load(path):
    return np.array(mx.load(path)["embedding"].astype(mx.float32)), str(mx.load(path)["embedding"].dtype)

(ref, ref_type), (other, other_type) = load(sys.argv[1]), load(sys.argv[2])
print("reference", ref.shape, ref_type, "| other", other.shape, other_type)
if ref.shape != other.shape:
    sys.exit("shapes differ")
diff = other - ref
norm = lambda a: np.sqrt((a.astype(np.float64) ** 2).sum(axis=-1))
row_cos = (ref.astype(np.float64) * other).sum(-1) / (norm(ref) * norm(other))
row_rel = norm(diff) / norm(ref)
ratio = norm(other) / norm(ref)
print("reference: mean %.4f std %.4f, row norm %.2f..%.2f" % (ref.mean(), ref.std(), norm(ref).min(), norm(ref).max()))
print("relative L2 error, whole matrix: %.4f" % (norm(diff.ravel()) / norm(ref.ravel())))
print("cosine, whole matrix: %.6f" % ((ref.astype(np.float64) * other).sum() / (norm(ref.ravel()) * norm(other.ravel()))))
print("cosine per token: min %.6f (token %d), median %.6f, mean %.6f" % (row_cos.min(), row_cos.argmin(), np.median(row_cos), row_cos.mean()))
print("relative L2 per token: max %.4f (token %d), median %.4f" % (row_rel.max(), row_rel.argmax(), np.median(row_rel)))
print("row norm ratio other/reference: %.4f..%.4f" % (ratio.min(), ratio.max()))
print("max abs difference %.4f, at reference value %.4f" % (np.abs(diff).max(), ref.ravel()[np.abs(diff).argmax()]))
quarters = np.array_split(row_rel, 4)
print("relative L2 per token, by quarter of the sequence:", " ".join("%.4f" % q.mean() for q in quarters))
