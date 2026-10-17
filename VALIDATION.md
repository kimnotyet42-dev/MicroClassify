# Reproducibility validation

The validation covers the experiments reported in `../main.tex` on EMDS-7
(2,365 images, 41 classes), the self-collected microscopy dataset (390 images,
12 folders), and ZooLake2 (29,499 images, 35 classes).

## Manuscript results covered

The reference values in `expected_paper_values.json` cover:

- the EMDS-7 FSCIL AA/PD comparison, base-class shot study, and all ten
  per-split AA/PD pairs;
- the self-collected golden-mussel and *Daphnia magna* automatic/oracle F1
  values;
- the EMDS-7 controlled target-proportion results and their paired and
  classwise diagnostics;
- the EMDS-7 all-class-support classifier comparison and the MicroClassify row
  evaluated on the same query sets;
- the EMDS-7 and ZooLake2 module ablations and DINOv2/DINOv3/PE-Core
  comparisons;
- all six ZooLake2 shot-curve means and population standard deviations;
- all eight neighborhood-size and four bandwidth-sensitivity results on both
  EMDS-7 and ZooLake2; and
- the effective-resolution accuracy and entropy values at 224, 128, 96, 64,
  48, and 32 pixels.

Use `python tools/validate_paper_results.py` to compare generated results with
these references. See README.md for the CLIP reference-environment option.

The EMDS-7 full-whitening result at 224 pixels is 78.346% before rounding; the
manuscript reports 78.4%.

## Controlled target-proportion results

The EMDS-7 protocol contains 19 target classes, ten support sets per class, and
200 query images per task. At target proportions of 20%, 10%, 5%, and 2.5%, the
automatic-threshold F1 medians are 73.9%, 71.0%, 58.8%, and 37.8%; the oracle
medians are 82.8%, 81.7%, 76.6%, and 66.7%.

From 5% to 2.5%, the paired median automatic-threshold change is -17.1
percentage points, and 91.6% of the 190 target-class--seed pairs decline. On
the common query sets used for the all-class-support comparison, MicroClassify
gives 92.3%, 81.0%, 79.6%, 68.3%, 69.4%, 64.8%, and 61.8% for class-pool sizes
5, 10, 15, 20, 25, 30, and 35.

## Graph-parameter sensitivity

For EMDS-7 under the 41-way 4-shot protocol, mean accuracies at
`k=1/3/5/10/15/20/30/50` are
17.37/64.03/69.98/71.14/72.64/70.87/67.72/65.39%. With `k=10`, the local
geometric mean, kth-nearest-neighbor distance, local median, and global median
bandwidth estimators give 71.14%, 69.86%, 69.37%, and 69.10%, respectively.

The corresponding ZooLake2 reference values are provided in
`expected_paper_values.json`.

## CLIP-family reference environment

The OpenAI CLIP and BioCLIP manuscript values were generated on an RTX 3090
with Python 3.12.3, PyTorch 2.8.0+cu128, OpenCLIP 3.3.0, and CUDA autocast. The
checkpoint hashes are:

- OpenAI CLIP ViT-B/16:
  `5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f`
- BioCLIP v1:
  `e380384f0c30d425d8c6c40f24471f9dd497fbdfa734a89c461a94aee95f0ef4`

An RTX 3050 Laptop GPU with PyTorch 2.5.1+cu121 gives BioCLIP
logistic-regression F1 values of
86.6/76.5/74.1/64.1/60.9/57.1/53.4, compared with the manuscript values
86.6/76.6/74.1/63.9/60.9/57.1/53.4. OpenAI CLIP gives
80.4/73.1/65.0/56.3/51.9/47.3/45.7, compared with
81.8/73.1/65.0/56.4/52.2/47.5/45.7. The checkpoint files, class pools,
support/query splits, logistic-regression `C` grid, and aggregation rules are
identical. These small differences arise from CUDA-autocast feature extraction
across hardware and software environments.

## Runtime reference

On an RTX 3050 Laptop GPU, the 390-image workload takes approximately 344
seconds for feature extraction, 31.2 ms for Graph-z, and 722 ms for
within-class residual construction plus Ledoit--Wolf whitening-matrix
estimation. The 1,600-image workload takes approximately 504 seconds, 94.9 ms,
and 339.4 ms, respectively. Runtime values depend on the hardware and system
load; the code fixes the measured workload and timing scope.

## Feature-cache consistency

Feature caches record the ordered sample fingerprint, backbone, input
resolution, checkpoint path, and completed prefix. A cache is reused only when
these fields match the requested experiment. In the ZooLake2 cache checks,
DINOv2 and PE-Core features match element-for-element across repeated
extraction. DINOv3 features can differ at approximately `1e-6` across GPU
executions without changing the reported downstream results.
