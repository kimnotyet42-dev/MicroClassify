# Validation status

Validation was performed against `paper/myPaper4/main.tex` using the three
configured datasets: EMDS-7 (2,365 images, 41 classes), the self-collected
microscopy data (423 images, 18 folders), and ZooLake2 (29,499 images, 35
classes).

## Strictly reproduced manuscript results

- EMDS-7 FSCIL main AA/PD results, fixed/random base-class experiments, and all
  ten per-split AA/PD pairs.
- Self-collected golden-mussel and *Daphnia magna* automatic/oracle F1 values.
- All 42 EMDS-7 threshold-scan cells: seven class-pool sizes, three shots, and
  automatic/oracle thresholds.
- EMDS-7 and ZooLake2 module ablations and DINOv2/DINOv3/PE-Core comparisons.
- All six ZooLake2 shot-curve means and population standard deviations.
- All eight neighborhood-size and four bandwidth-sensitivity results.
- The effective-resolution trend and entropy values at 224, 128, 96, 64, 48,
  and 32 pixels. The local 224-pixel full-whitening run is 78.346%, whereas the
  manuscript reports the associated 78.4% reference value.
- DINOv3 logistic-regression and SimpleShot rows in the EMDS-7 classifier
  comparison.

The optimized distance reuse, multi-column graph propagation, self-collected
graph reuse, feature prefetching, and low-rank whitening paths were compared
against their retained computations on fixed inputs. Tests are in `tests/`.

## CLIP-family reference environment

The OpenAI CLIP and BioCLIP paper values were produced on an RTX 3090 with
Python 3.12.3, PyTorch 2.8.0+cu128, OpenCLIP 3.3.0, and CUDA autocast. The local
rerun used an RTX 3050 Laptop GPU and PyTorch 2.5.1+cu121. The official weight
hashes are:

- OpenAI CLIP ViT-B/16:
  `5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f`
- BioCLIP v1:
  `e380384f0c30d425d8c6c40f24471f9dd497fbdfa734a89c461a94aee95f0ef4`

Under the local environment, BioCLIP logistic-regression F1 values are
86.6/76.5/74.1/64.1/60.9/57.1/53.4, versus the manuscript reference
86.6/76.6/74.1/63.9/60.9/57.1/53.4. OpenAI CLIP gives
80.4/73.1/65.0/56.3/51.9/47.3/45.7, versus
81.8/73.1/65.0/56.4/52.2/47.5/45.7. The code, weight hashes, class pools,
support/query splits, C grid, and aggregation are identical; the remaining
difference is isolated to autocast feature extraction across environments.

## Runtime rerun

On the current RTX 3050 machine, the 420-image workload took 185.0 seconds for
feature extraction, 10.9 ms for Graph-z, and 302 ms for full whitening. The
1,600-image workload took 517.0 seconds, 99.9 ms, and 334 ms, respectively.
Runtime is hardware- and load-dependent and is validated by workload and timing
scope rather than exact wall-clock equality.

## Feature-cache provenance

EMDS-7, self-collected, CLIP-family, and effective-resolution features were
extracted from the configured raw images during this validation. Existing
ZooLake2 paper caches were imported only after recomputing their first reference
batch: DINOv2 and PE-Core matched element-for-element; DINOv3 differed at about
1e-6 across GPU executions. All ZooLake2 downstream paper results were then
recomputed from the retained paper caches.
