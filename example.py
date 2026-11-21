"""Run the codec on synthetic calibration queries and a separate test query."""

import numpy as np
import torch

from vector_codec import VectorCodec


def main():
    torch.set_num_threads(2)
    rng = np.random.default_rng(42)
    centers = rng.normal(0, 1 / np.sqrt(512), (64, 256, 8)).astype("float32")
    codec = VectorCodec(centers)
    gallery = rng.normal(size=(256, 512)).astype("float32")
    probes = rng.normal(size=(128, 512)).astype("float32")
    query = gallery[17] + rng.normal(0, 0.1, 512).astype("float32")
    plain = codec.encode(gallery, None, passes=0)
    codes = codec.encode(gallery, probes, initial_codes=plain)
    scores = codec.scores(query, codes)
    order = np.argsort(-scores, kind="stable")

    decoded = centers[np.arange(64), codes].reshape(-1, 512)
    expected = (decoded @ query) / (np.linalg.norm(decoded, axis=1) * np.linalg.norm(query))
    np.testing.assert_allclose(scores, expected, atol=2e-6, rtol=1e-5)
    assert codes.dtype == np.uint8 and codes.nbytes == 64 * len(gallery)
    assert np.all((plain != codes).sum(1) <= 2)
    print(f"{len(codes)} vectors, {codes.nbytes:,} bytes of codes, {centers.nbytes:,} shared codebook bytes")
    print(f"Rows changed during refinement: {np.any(plain != codes, axis=1).sum()}")
    print(f"Test query based on row 17; nearest gallery rows: {order[:5].tolist()}")


if __name__ == "__main__":
    main()
