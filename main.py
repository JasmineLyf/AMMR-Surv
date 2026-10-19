import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import sklearn
import torch
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.nn import functional as F

from data import Cohort, prepare, save_json
from evaluation import evaluate, survival_at
from model import AMMRSurv, augment, auxiliary_loss, prototype_loss, survival_loss


SEEDS = [42, 73, 101, 137, 211]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def precision(device, enabled):
    return torch.autocast(device_type="cuda", dtype=torch.float16) if enabled and device.type == "cuda" else contextlib.nullcontext()


def optimizer_for(model):
    groups = {}
    for name, parameter in model.named_parameters():
        encoder = name.startswith(("pet.", "ct."))
        decay = parameter.ndim > 1 and not name.endswith("bias")
        key = (encoder, decay)
        groups.setdefault(key, []).append(parameter)
    return torch.optim.AdamW([
        {"params": parameters, "lr": 1e-4 if encoder else 3e-4, "weight_decay": 1e-4 if decay else 0.0}
        for (encoder, decay), parameters in groups.items()
    ], betas=(0.9, 0.999), eps=1e-8)


def train_epoch(model, cohort, indices, transforms, boundaries, optimizer, scaler, device, warmup, amp):
    model.train()
    order = np.random.permutation(indices)
    totals = np.zeros(3)
    for start in range(0, len(order), 4):
        group = order[start:start + 4]
        evaluable = 0
        for index in group:
            _, targets, events = cohort.metadata(index)
            valid = np.isfinite(targets) & np.array(transforms["target_enabled"])[None] & (events == 0)[:, None]
            evaluable += int(valid.any())
        optimizer.zero_grad(set_to_none=True)
        for index in group:
            patient = augment(cohort.load(index, transforms, device))
            with precision(device, amp):
                output = model(patient, warmup=warmup)
                ls = survival_loss(output["logits"], patient["time"], patient["outcome"], boundaries)
                lm, valid = auxiliary_loss(output, patient)
                lp = output["response"].sum() * 0 if warmup else prototype_loss(output, model.prototypes)
                loss = (ls + 0.05 * lp) / len(group)
                if valid:
                    loss = loss + 0.20 * lm / evaluable
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite training loss")
            scaler.scale(loss).backward()
            totals += np.array([ls.item(), lm.item(), lp.item()])
            del patient, output, loss
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
        scaler.step(optimizer)
        scaler.update()
    return (totals / len(order)).tolist()


@torch.no_grad()
def validation_loss(model, cohort, indices, transforms, boundaries, device, amp):
    model.eval()
    losses = []
    for index in indices:
        patient = cohort.load(index, transforms, device)
        with precision(device, amp):
            output = model(patient)
        losses.append(survival_loss(output["logits"], patient["time"], patient["outcome"], boundaries).item())
    result = float(np.mean(losses))
    if not np.isfinite(result):
        raise FloatingPointError("Nonfinite validation loss")
    return result


@torch.no_grad()
def initialize_prototypes(model, cohort, indices, transforms, device, seed, amp):
    model.eval()
    embeddings = []
    for index in indices:
        patient = cohort.load(index, transforms, device)
        with precision(device, amp):
            output = model(patient, warmup=True)
        embeddings.append(F.normalize(output["response"], dim=-1).cpu().numpy())
    embeddings = np.concatenate(embeddings)
    if len(embeddings) < 4:
        raise ValueError("At least four fitting lesions are required")
    clustering = KMeans(n_clusters=4, n_init=20, max_iter=300, tol=1e-4, algorithm="lloyd", random_state=seed).fit(embeddings)
    if len(np.unique(clustering.labels_)) < 4:
        raise ValueError("Fitting embeddings cannot initialize four distinct prototypes")
    centers = torch.tensor(clustering.cluster_centers_, device=device, dtype=torch.float32)
    model.prototypes.copy_(F.normalize(centers, dim=-1))


def save_checkpoint(path, model, metadata):
    temporary = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), **metadata}, temporary)
    temporary.replace(path)


def run_fold(cohort, fitting, validation, testing, args, repeat, fold, output, grid):
    seed = SEEDS[repeat] * 100 + fold
    seed_everything(seed)
    transforms = cohort.fit_transforms(fitting)
    events = [cohort.rows[i]["time"] for i in fitting if cohort.rows[i]["event"]]
    boundaries = np.quantile(events, [0.2, 0.4, 0.6, 0.8, 1.0], method="linear")
    if np.any(np.diff(np.r_[0, boundaries]) <= 0):
        raise ValueError("Event quantiles are not strictly increasing; five intervals cannot be identified")
    if boundaries[-1] < 24:
        raise ValueError("Fitting event-time support ends before 24 months; no extrapolation is allowed")
    metadata = {"repeat": repeat + 1, "fold": fold + 1, "seed": seed,
                "boundaries": boundaries.tolist(), "transforms": transforms,
                "fitting": [cohort.rows[i]["patient_id"] for i in fitting],
                "validation": [cohort.rows[i]["patient_id"] for i in validation],
                "testing": [cohort.rows[i]["patient_id"] for i in testing]}
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "split.json", metadata)
    device = torch.device(args.device)
    model = AMMRSurv(checkpoint_encoders=not args.no_checkpoint).to(device)
    optimizer = optimizer_for(model)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    log_path = output / "training.jsonl"
    with log_path.open("w") as log:
        for epoch in range(args.warmup_epochs):
            losses = train_epoch(model, cohort, fitting, transforms, boundaries, optimizer, scaler, device, True, args.amp)
            record = {"phase": "warmup", "epoch": epoch + 1, "losses_surv_met_proto": losses}
            log.write(json.dumps(record) + "\n")
            log.flush()
            print(f"repeat={repeat + 1} fold={fold + 1} warmup={epoch + 1} nll={losses[0]:.6f}", flush=True)
        initialize_prototypes(model, cohort, fitting, transforms, device, seed, args.amp)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.01 + 0.99 * (1 + math.cos(math.pi * epoch / args.epochs)) / 2)
        best, significant_best, stale = float("inf"), float("inf"), 0
        for epoch in range(args.epochs):
            learning_rates = [group["lr"] for group in optimizer.param_groups]
            losses = train_epoch(model, cohort, fitting, transforms, boundaries, optimizer, scaler, device, False, args.amp)
            nll = validation_loss(model, cohort, validation, transforms, boundaries, device, args.amp)
            scheduler.step()
            if nll < best:
                best = nll
                save_checkpoint(output / "best.pt", model, {**metadata, "epoch": epoch + 1, "validation_nll": best})
            if nll < significant_best - 1e-4:
                significant_best, stale = nll, 0
            else:
                stale += 1
            log.write(json.dumps({"phase": "joint", "epoch": epoch + 1, "losses_surv_met_proto": losses,
                                  "validation_nll": nll, "learning_rates": learning_rates}) + "\n")
            log.flush()
            print(f"repeat={repeat + 1} fold={fold + 1} joint={epoch + 1} validation_nll={nll:.6f}", flush=True)
            if stale >= args.patience:
                break
    state = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    predictions, logits = [], []
    with torch.no_grad():
        for index in testing:
            patient = cohort.load(index, transforms, device)
            with precision(device, args.amp):
                result = model(patient)
            values = result["logits"].cpu().numpy()
            logits.append(values)
            predictions.append(survival_at(values, boundaries, grid))
    temporary = output / "predictions.tmp.npz"
    np.savez_compressed(temporary, indices=np.array(testing), survival=np.stack(predictions),
                        logits=np.stack(logits), boundaries=boundaries, grid=grid)
    temporary.replace(output / "predictions.npz")
    del model, optimizer, scaler, state
    if device.type == "cuda":
        torch.cuda.empty_cache()


def fingerprint(cohort, args):
    content = vars(args).copy()
    for key in ("resume", "output", "manifest"):
        content.pop(key, None)
    content["torch"] = torch.__version__
    content["numpy"] = np.__version__
    content["sklearn"] = sklearn.__version__
    content["python"] = sys.version
    content["seeds"] = SEEDS
    content["files"] = {}
    paths = [Path(args.manifest)] + [Path(__file__).parent / name for name in ("main.py", "model.py", "data.py", "evaluation.py")]
    paths += [cohort.root / row["file"] for row in cohort.rows]
    for path in paths:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        content["files"][path.name] = digest.hexdigest()
    return content


def train(args):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; install a CUDA-enabled PyTorch build or pass --device cpu")
    torch.set_num_threads(args.threads)
    cohort = Cohort(args.manifest)
    if args.expected_patients and len(cohort.rows) != args.expected_patients:
        raise ValueError(f"Expected {args.expected_patients} patients, found {len(cohort.rows)}")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = fingerprint(cohort, args)
    config_path = output / "run_config.json"
    if config_path.exists():
        if not args.resume:
            raise FileExistsError("Output exists; use --resume to skip completed folds")
        if json.loads(config_path.read_text()) != config:
            raise ValueError("Code, data, environment, or arguments changed; use a new output directory")
    elif any(output.iterdir()):
        raise FileExistsError("Output must be empty for a new run")
    save_json(config_path, config)
    event = np.array([row["event"] for row in cohort.rows])
    time = np.array([row["time"] for row in cohort.rows])
    grid = np.arange(0, 36.0 + 0.125, 0.25)
    curves = np.full((5, len(cohort.rows), len(grid)), np.nan)
    assigned = np.zeros((5, len(cohort.rows)), dtype=int)
    for repeat, seed in enumerate(SEEDS):
        splitter = StratifiedKFold(5, shuffle=True, random_state=seed)
        for fold, (outer, testing) in enumerate(splitter.split(np.zeros(len(event)), event)):
            fitting, validation = train_test_split(outer, test_size=0.2, stratify=event[outer], random_state=seed * 100 + fold)
            fold_output = output / f"repeat_{repeat + 1}" / f"fold_{fold + 1}"
            prediction_file = fold_output / "predictions.npz"
            if not (args.resume and prediction_file.exists()):
                run_fold(cohort, fitting, validation, testing, args, repeat, fold, fold_output, grid)
            with np.load(prediction_file, allow_pickle=False) as data:
                if not np.array_equal(data["indices"], testing) or not np.array_equal(data["grid"], grid):
                    raise ValueError("Saved fold predictions do not match the split")
                curves[repeat, testing] = data["survival"]
                assigned[repeat, testing] += 1
    if not np.all(assigned == 1):
        raise RuntimeError("Every patient must have exactly one OOF prediction per repetition")
    aggregated = curves.mean(axis=0)
    np.savez_compressed(output / "oof_predictions.npz", survival_by_repeat=curves, survival=aggregated,
                        grid=grid, time=time, event=event, patient_ids=np.array([row["patient_id"] for row in cohort.rows]))
    end24 = int(np.flatnonzero(grid == 24)[0])
    evaluation = evaluate(time, event, aggregated[:, :end24 + 1], grid[:end24 + 1], args.bootstrap)
    save_json(output / "metrics.json", evaluation)
    import csv
    with (output / "oof_risk24.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["patient_id", "landmark_os_months", "event", "oof_risk24"])
        writer.writerows((row["patient_id"], row["time"], row["event"], float(1 - aggregated[i, end24]))
                         for i, row in enumerate(cohort.rows))
    print(json.dumps(evaluation, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare")
    preparation.add_argument("--patients", required=True)
    preparation.add_argument("--sites", required=True)
    preparation.add_argument("--output", required=True)
    training = commands.add_parser("train")
    training.add_argument("--manifest", required=True)
    training.add_argument("--output", required=True)
    training.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    training.add_argument("--warmup-epochs", type=int, default=20)
    training.add_argument("--epochs", type=int, default=150)
    training.add_argument("--patience", type=int, default=25)
    training.add_argument("--bootstrap", type=int, default=2000)
    training.add_argument("--expected-patients", type=int, default=126)
    training.add_argument("--threads", type=int, default=8)
    training.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    training.add_argument("--no-checkpoint", action="store_true")
    training.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.patients, args.sites, args.output)
    else:
        if min(args.warmup_epochs, args.epochs, args.patience, args.threads) < 1 or args.bootstrap < 0:
            parser.error("Epochs, patience, threads must be positive; bootstrap must be nonnegative")
        train(args)


if __name__ == "__main__":
    main()
