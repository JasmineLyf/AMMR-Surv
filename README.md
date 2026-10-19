# AMMR-Surv: Main Experiments

A complete implementation of AMMR-Surv, covering audited longitudinal lesion inputs, paired cropping, model training, five repeats of five-fold patient-level cross-validation, aggregation of out-of-fold (OOF) survival curves, and evaluation using four primary metrics.

## Running the Experiments

Navigate to this directory in a Linux environment with an NVIDIA GPU, or another environment capable of running PyTorch. Python 3.12 is recommended. First, install a PyTorch CUDA build compatible with your GPU driver, then install the remaining dependencies. The `requirements.txt` file does not enforce a specific CUDA version.

```bash
cd /path/to/ammr_main
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c 'import torch; print(torch.__version__); print(torch.cuda.is_available())'
python main.py prepare --patients /path/to/patients.csv --sites /path/to/sites.csv --output data/crops
python main.py train --manifest data/crops/manifest.json --output runs/ammr_main
python -m pip freeze > runs/ammr_main/environment.txt
```

By default, the full experiment trains 25 models. Each model undergoes 20 warm-up epochs, followed by up to 150 joint-training epochs, with early stopping based on validation performance. If CUDA is unavailable, the program exits with an explicit error. CPU execution is supported via `--device cpu`, but a GPU is recommended for the full experiment.

To resume an interrupted run:

```bash
python main.py train --manifest data/crops/manifest.json --output runs/ammr_main --resume
```

The `--resume` option skips folds whose prediction exports are complete. Incomplete folds are retrained from scratch using the same random seed; training does not resume from an intermediate epoch.

Fingerprint checks verify the code, input files, run parameters, and key dependency versions. Use a new output directory if any of these have changed. The `prepare` command does not overwrite an existing manifest.

## Outputs

| File | Contents |
| --- | --- |
| `run_config.json` | Command-line arguments, dependency versions, random seeds, and SHA-256 hashes of the code and inputs |
| `repeat_R/fold_F/split.json` | Patient assignments to the fitting, validation, and test sets; fold-specific statistics; and survival time-bin boundaries |
| `repeat_R/fold_F/training.jsonl` | Per-epoch training loss, validation negative log-likelihood (NLL), and learning rate |
| `repeat_R/fold_F/best.pt` | Model checkpoint with the lowest validation survival NLL, together with fold-specific transformation parameters |
| `repeat_R/fold_F/predictions.npz` | Outer-fold test predictions, logits, and fold-specific time-bin boundaries |
| `oof_predictions.npz` | OOF survival curves from all five repeats and their mean survival probabilities on a shared 0–36-month time grid |
| `oof_risk24.csv` | Patient-level OOF 24-month mortality probabilities and observed outcomes |
| `metrics.json` | Uno’s C-index at 24 months, Harrell’s C-index, AUC at 24 months, and the integrated Brier score (IBS) over 0–24 months, with bootstrap confidence intervals |
