# Two-pass vector compression

Store a 512-dimensional vector in 64 bytes and search the codes by cosine similarity. At index-build time, two refinement passes use representative queries to choose which product-quantization symbols to change. The query path stays the same as for ordinary cosine-scored product codes.

将 512 维向量写成 64 字节的短码，用于余弦相似度检索。建库时多做两轮计算，目标是在相同短码长度下保留更多原始近邻。适合已有向量、能取得历史查询样本、希望缩小候选索引的团队试用。

## Run the example

With Python 3.10 or newer, run these commands in the repository directory.

```sh
python -m pip install -r requirements.txt
python example.py
```

The example generates a codebook, 256 gallery vectors, calibration queries, and one separate test query. It checks the code size and compares the lookup-table scores with a direct cosine calculation. The gallery codes occupy 16,384 bytes. The shared codebook occupies another 524,288 bytes.

## Use your own vectors

A codebook has shape `(64, 256, 8)`. Each of the 64 blocks has 256 possible centers, so one byte identifies the center for that block. Use a codebook fitted on representative, unit-length gallery vectors. For an OPQ codebook, apply its orthogonal rotation to all gallery vectors and queries before calling the codec. Keep that rotation with the codebook.

```python
import numpy as np
from vector_codec import VectorCodec

codec = VectorCodec(centers)
codes = codec.encode(gallery, calibration_queries)  # (N, 512) and (P, 512), P >= 32
np.save("gallery_codes.npy", codes, allow_pickle=False)

scores = codec.scores(query, codes)  # query.shape == (512,)
top10 = np.argsort(-scores, kind="stable")[:10]
```

The encoder starts with the nearest center in each block. Each pass tries one alternative per block and accepts at most one change per vector. It chooses the change that best preserves cosine scores on the 32 closest calibration queries, with a small penalty for reconstruction error. Use `passes=0` for the ordinary PQ baseline, or supply existing 64-byte PQ codes with `initial_codes=`. Pass `device="cuda"` for GPU encoding. The reference scanner uses NumPy on CPU and scans the gallery once per query. Calibration queries are needed only when building codes.

To fit a plain PQ codebook from scratch, install the optional `faiss-cpu` package. In this example, `training` contains representative nonzero 512-D gallery vectors. Use at least 256 training rows and preferably several thousand more.

```python
import faiss
import numpy as np

training = np.array(training, dtype=np.float32, order="C", copy=True)
faiss.normalize_L2(training)
pq = faiss.ProductQuantizer(512, 64, 8)
pq.train(training)
centers = faiss.vector_to_array(pq.centroids).reshape(64, 256, 8)
```

## Catalog retrieval result

In a public fashion catalog, 8,000 held-out product photos were used to search 30,567 distinct titles. Both encoders used the same catalog-fitted OPQ codebook and the same scan procedure. The two-pass encoder used 4,480 separate calibration queries. Each gallery entry occupied 64 bytes under either method.

| Measure | Ordinary PQ | Two-pass PQ |
| --- | ---: | ---: |
| Original top-5 neighbors retained in compressed top-10 | 67.34% | 69.95% |
| Correct title at rank 1, out of 8,000 | 230 | 241 |
| Correct title in top 10, out of 8,000 | 1,237 | 1,267 |

The clearest gain was in preserving the uncompressed ranking. The confidence interval for the change in correct-title top-1 included zero. Index construction takes extra work. Total memory also includes the codebook, any rotation, item identifiers and search buffers. The figures above cover one catalog. The synthetic example verifies the codec mechanics separately. [Catalog dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small)

## License and commercial inquiries

This source is available under the PolyForm Small Business License 1.0.0. Qualifying companies can use it without a license fee. The license sets two limits, both of which must be met. The company must have fewer than 100 employees and independent contractors combined, and its prior tax year's total revenue must fall below the inflation-adjusted equivalent of USD 1 million in 2019. The definition of company includes organizations under common control. [Full license](LICENSE)

For use outside these terms, request a separate commercial license from Lishu-Technologies. Open an issue titled **Commercial licensing inquiry** to reach the maintainers. A brief description of the intended use is sufficient. [Contact the maintainers](https://github.com/Lishu-Technologies/two-pass-vector-compression/issues/new)
