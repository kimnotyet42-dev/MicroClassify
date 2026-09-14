# MicroClassify reproducibility code

This directory reproduces the experiments retained in the reduced manuscript:
EMDS-7, the self-collected microscopy dataset, and ZooLake2. Deleted pathology,
natural-image, graph-size, and exploratory experiments are intentionally absent.

Start with [SETUP.md](SETUP.md). A reproducer edits only
`configs/paths.ini`; paper protocols remain fixed in the experiment code.

## Structure

```text
configs/       Machine-specific path template
dataio/        Dataset readers and deterministic evaluation protocols
methods/       Frozen backbones, whitening, graph propagation, thresholds,
               feature caching, baselines, and result writing
experiments/   One entry point for each retained experimental question
plots/         Figures generated from experiment CSV files
tests/         Protocol and numerical-equivalence checks
results/       Generated raw and summary CSV files; ignored by Git
cache/         Resumable feature caches; ignored by Git
figures/       Generated PDF figures; ignored by Git
```

## Paper experiments

All commands are run from this directory. Add `--resume` to continue a partial
experiment.

### EMDS-7 FSCIL, base-shot study, session curve, and split results

```bash
python -m experiments.emds7_fscil --resume
python -m plots.plot_fscil_sessions
```

Outputs: `results/emds7_fscil.csv`, `emds7_fscil_sessions.csv`,
`emds7_fscil_splits.csv`, and `figures/sessions.pdf`.

### Self-collected binary results and EMDS-7 one-vs-rest scan

```bash
python -m experiments.threshold_ablation --suite all --workers 8 --resume
```

This also evaluates direct Otsu, equal-weight KDE + Otsu, weighted KDE + Otsu,
and the oracle threshold on exactly the same decision scores.

### EMDS-7 comparison with all-class-support classifiers

```bash
python -m experiments.binary_model_comparison --cv-jobs 1 --resume
```

The paper rows are DINOv3 + SimpleShot, DINOv3 + logistic regression,
BioCLIP + logistic regression, and OpenAI CLIP + logistic regression.

### ZooLake2 main table, module ablation, and cross-backbone table

```bash
python -m experiments.module_backbone_ablation \
  --datasets emds7 zoolake2 \
  --backbones dinov2 dinov3 pecore \
  --resume
```

The DINOv3/ZooLake2 row also supplies the NC, Graph-z, and whitening + Graph-z
values used by the main ZooLake2 table. The fully supervised MobileNet row is a
cited result from Chen et al. (2025), not an experiment reproduced here.

### ZooLake2 labeled-shot curve

```bash
python -m experiments.zoolake2_shot_curve --resume
python -m plots.plot_zoolake2_shot_curve
```

The curve uses the fixed 31-class protocol, 15 queries per class, and 100
episodes for each of 4, 8, 12, 16, 20, and 24 shots.

### Runtime table

```bash
python -m experiments.runtime_benchmark
```

Runtime is hardware dependent. This command measures the published workload;
exact wall-clock values are expected only on equivalent hardware and software.

### EMDS-7 effective-resolution experiment

```bash
python -m experiments.effective_resolution --resume
python -m plots.plot_effective_resolution
```

### ZooLake2 graph-parameter sensitivity

```bash
python -m experiments.graph_parameter_sensitivity --analysis all --resume
```

Each episode computes its pairwise distance matrix once and reuses it for all
neighborhood and bandwidth settings. Raw per-episode values and a summary CSV
are written incrementally.

## Run everything

```bash
python reproduce_all.py
```

Useful alternatives:

```bash
python reproduce_all.py --skip-runtime
python reproduce_all.py --only emds7_fscil
python reproduce_all.py --config configs/server.ini
```

The machine-readable mapping from manuscript outputs to commands is stored in
`experiment_manifest.json`.

After the runs finish, compare every generated manuscript value with the
checked-in reference table:

```bash
python tools/validate_paper_results.py
```

On hardware other than the RTX 3090 CLIP reference environment, use
`--allow-reference-environment-drift` to keep the documented CUDA-autocast
differences as warnings while still enforcing every other manuscript value.

## Reproducibility rule

Optimizations may reuse feature caches, distance matrices, graph transitions,
or label columns, but must not change dataset eligibility, support/query
membership, random seeds, image preprocessing, model precision, metrics, or
aggregation. A migrated experiment is accepted only after its raw outputs and
rounded manuscript values agree with the retained reference implementation.

OpenAI CLIP and BioCLIP use CUDA autocast, matching the original comparison
run. Their few-shot predictions can change slightly across GPU and PyTorch
versions when a query lies close to a decision boundary. The manuscript values
were produced with Python 3.12.3, PyTorch 2.8.0+cu128, OpenCLIP 3.3.0, and an
RTX 3090. `VALIDATION.md` records both that reference and the local RTX 3050
rerun rather than silently treating them as identical.

## Current verification status

The repository has been migrated to the shared `methods/` and `dataio/`
interfaces. Static, import, protocol, numerical-equivalence, and manuscript
result checks have been run locally. See `VALIDATION.md` for exact matches,
hardware-sensitive results, and cache provenance.
