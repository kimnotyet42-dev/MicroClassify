# Environment and local paths

The source uses Python 3.8-compatible syntax. Python 3.9 or newer is
recommended because the supported versions of PyTorch and Transformers may no
longer install on older Python releases.

## 1. Install dependencies

Create an isolated environment and install a CUDA build of PyTorch compatible
with the local machine. For CUDA 12.1, for example:

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

For the exact Python 3.11/CUDA 12.1 package versions used by the local reference
environment, use `requirements-lock-cu121.txt` instead. The broader
`requirements.txt` is intended for older compatible Python environments.

The CLIP-family comparison values in the manuscript were generated on an RTX
3090 with Python 3.12.3, PyTorch 2.8.0+cu128, and OpenCLIP 3.3.0. CUDA autocast
can move a few boundary predictions on different hardware; see `VALIDATION.md`.

CPU execution is supported for post-processing. A CUDA device is
recommended when image features must be extracted from scratch.

## 2. Configure data and model paths

Copy the committed example file:

```text
configs/paths.example.ini -> configs/paths.ini
```

Edit only `configs/paths.ini`. Each dataset entry points directly to the
directory containing that dataset; model entries point to a local checkpoint
file or model directory. Relative cache, result, and figure paths are resolved
from this `code` directory.

Windows users should use forward slashes in the INI file, for example:

```ini
[datasets]
emds7 = D:/datasets/EMDS7/EMDS7
self_collected = D:/datasets/Microscopy
zoolake2 = D:/datasets/ZooLake2
```

The local `configs/paths.ini` is ignored by Git.

## 3. Limit numerical-library threads

The EMDS-7 threshold and logistic-regression experiments may start several
workers. On a machine or container with a CPU-thread limit, set the following
variables before running them:

```powershell
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:NUMEXPR_NUM_THREADS = '1'
```

## 4. Check the installation

From this directory, run:

```bash
python reproduce_all.py --help
python -m experiments.golden_mussel --help
python -m experiments.daphnia --help
```

Feature caches contain an ordered sample fingerprint, backbone name, input
resolution, checkpoint path, and completed prefix. A partial cache is reused
only when these fields match.
