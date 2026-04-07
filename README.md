# CAFormer

Minimal public release for the CAFormer RSVP classification model.

This release keeps only the code needed to train and evaluate the proposed
CAFormer model. Baseline implementations, reviewer-specific experiment scripts,
local results, checkpoints, logs, and datasets are intentionally excluded.

## Repository Structure

- `main.py`: YAML-driven training and evaluation entry point.
- `configs/`: full CAFormer configs for `THU`, `CAS`, and `GIST`.
- `Models/ablation_study/CAFormer.py`: CAFormer architecture.
- `Deep_learning/rsvp_classification_with_CSSL.py`: CAFormer trainer with the CSSL branch.
- `Data_Processing/make_dataset.py`: DataLoader helper for prepared fold files.
- `project_utils/naming.py`: lightweight model and result naming helpers.

## Data Layout

Prepare fold files before running experiments. The expected layout is:

```text
Dataset/<DATASET>/fold5_data/sub*.npz
```

Each `.npz` file should contain:

- `train_data`
- `train_label`
- `test_data`
- `test_label`
- `fold_length`

The expected EEG tensor shape before loading is `(N, C, T)`. The loader will
add the singleton input-channel dimension used by the model.

## Environment

Install PyTorch separately according to your CUDA / platform setup, then install
the remaining dependencies:

```powershell
pip install -r requirements.txt
```

The original experiments were run with CUDA-enabled PyTorch on an NVIDIA RTX
3090 GPU.

## Run CAFormer

THU:

```powershell
python main.py --config .\configs\caformer_thu.yaml
```

CAS:

```powershell
python main.py --config .\configs\caformer_cas.yaml
```

GIST:

```powershell
python main.py --config .\configs\caformer_gist.yaml
```

For a quick smoke test on a subset, override the subject count:

```powershell
python main.py --config .\configs\caformer_thu.yaml --subject_limit 3
```

## Outputs

Training checkpoints are written to:

```text
checkpoints/5fold_CAFormer_<DATASET>/
```

Results are written to:

```text
results/5fold_CAFormer_<DATASET>/
```

These folders are ignored by Git.
