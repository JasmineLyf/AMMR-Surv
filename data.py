import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import torch


EVENTS = {"matched": 0, "disappeared": 1, "new": 2}
ANATOMY = {"primary": 0, "nodal": 1, "other": 2, "unknown": 3}


def number(value):
    try:
        result = float(value)
        return result if np.isfinite(result) else np.nan
    except (TypeError, ValueError):
        return np.nan


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def save_json(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(content, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def quantitative_targets(row):
    targets = np.full(4, np.nan, dtype=np.float32)
    if row["event"] != "matched" or row.get("quantitative_reliable") != "1":
        return targets
    pre = np.array([number(row.get("pre_" + key)) for key in ("suvmax", "mtv_ml", "tlg")])
    post = np.array([number(row.get("post_" + key)) for key in ("suvmax", "mtv_ml", "tlg")])
    valid = np.isfinite(pre) & np.isfinite(post) & (pre >= 0) & (post >= 0)
    if valid[0]:
        targets[0] = post[0] - pre[0]
    for index in range(3):
        if valid[index] and pre[index] > 0:
            targets[index + 1] = 100 * (post[index] - pre[index]) / pre[index]
    return targets


def prepare(patients_csv, sites_csv, output):
    import SimpleITK as sitk

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        raise FileExistsError("Choose an empty output directory; manifest.json already exists")
    patients = read_csv(patients_csv)
    sites = read_csv(sites_csv)
    root = Path(sites_csv).resolve().parent
    patient_ids = [p["patient_id"] for p in patients]
    if len(patient_ids) != len(set(patient_ids)):
        raise ValueError("Duplicate patient_id")
    grouped = {pid: [] for pid in patient_ids}
    for row in sites:
        if row["patient_id"] not in grouped:
            raise ValueError("Site belongs to a patient outside the cohort")
        if row.get("reviewed") != "1":
            raise ValueError("Every site must have reviewed=1 after expert adjudication")
        if row["event"] not in EVENTS or row["anatomy"] not in ANATOMY:
            raise ValueError("Invalid event or anatomy category")
        grouped[row["patient_id"]].append(row)

    def load(path):
        if not path:
            return None
        image = sitk.ReadImage(str(root / path))
        if image.GetDimension() != 3:
            raise ValueError("Only 3D NIfTI images are accepted")
        return image

    def corners(mask):
        binary = sitk.Cast(mask > 0, sitk.sitkUInt8)
        stats = sitk.LabelShapeStatisticsImageFilter()
        stats.Execute(binary)
        if not stats.HasLabel(1):
            return []
        box = stats.GetBoundingBox(1)
        return [mask.TransformIndexToPhysicalPoint(tuple(int(v) for v in corner))
                for corner in itertools.product(*[(box[d], box[d] + box[d + 3] - 1) for d in range(3)])]

    def crop(image, center, size, spacing, padding):
        reference = sitk.Image([size] * 3, sitk.sitkFloat32)
        reference.SetSpacing([spacing] * 3)
        reference.SetOrigin((center - (size - 1) * spacing / 2).tolist())
        result = sitk.Resample(image, reference, sitk.Transform(), sitk.sitkLinear, padding, sitk.sitkFloat32)
        array = sitk.GetArrayFromImage(result)
        if not np.isfinite(array).all():
            raise ValueError("Nonfinite image intensity")
        return array

    manifest = []
    for patient in patients:
        pid = patient["patient_id"]
        rows = grouped[pid]
        if not rows or len({row["site_id"] for row in rows}) != len(rows):
            raise ValueError(f"No sites or duplicate site IDs: {pid}")
        if all(row["event"] == "new" for row in rows):
            raise ValueError(f"At least one accepted baseline lesion is required: {pid}")
        duration, event = number(patient["landmark_os_days"]), number(patient["os_event"])
        if not np.isfinite(duration) or duration <= 0 or event not in (0, 1):
            raise ValueError(f"Invalid landmark survival: {pid}")
        images = {"pet": [], "ct": []}
        for row in rows:
            masks = [load(row.get(key, "")) for key in ("pre_mask", "post_mask")]
            mask_corners = [corners(mask) if mask is not None else [] for mask in masks]
            present = tuple(bool(points) for points in mask_corners)
            expected = {"matched": (True, True), "disappeared": (True, False), "new": (False, True)}
            if present != expected[row["event"]]:
                raise ValueError(f"Mask/event inconsistency: {pid}/{row['site_id']}")
            coordinates = np.asarray(mask_corners[0] + mask_corners[1])
            center = (coordinates.min(0) + coordinates.max(0)) / 2
            for modality, size, spacing, padding in (("pet", 64, 3.0, 0.0), ("ct", 96, 2.0, -1000.0)):
                pair = []
                for timepoint in ("pre", "post"):
                    image = load(row[timepoint + "_" + modality])
                    if image is None:
                        raise ValueError("Both actual image timepoints are required for every site")
                    center_index = image.TransformPhysicalPointToContinuousIndex(center.tolist())
                    if any(v < -0.5 or v > size_axis - 0.5 for v, size_axis in zip(center_index, image.GetSize())):
                        raise ValueError(f"Site center lies outside an image: {pid}/{row['site_id']}")
                    array = crop(image, center, size, spacing, padding)
                    if modality == "pet":
                        array = np.log1p(np.clip(array, 0, 25)) / np.log(26)
                    else:
                        array = np.clip(array, -1000, 1000) / 1000
                    pair.append(array[None].astype(np.float32))
                images[modality].append(np.stack(pair))
        filename = hashlib.sha256(pid.encode()).hexdigest()[:24] + ".npz"
        timing = np.array([number(patient.get(key)) for key in
                           ("prepost_interval_days", "post_minus_rt_end_days")], dtype=np.float32)
        np.savez_compressed(output / filename, pet=np.stack(images["pet"]), ct=np.stack(images["ct"]),
                            event=np.array([EVENTS[r["event"]] for r in rows], dtype=np.int64),
                            anatomy=np.array([ANATOMY[r["anatomy"]] for r in rows], dtype=np.int64),
                            timing=timing, targets=np.stack([quantitative_targets(r) for r in rows]),
                            site_ids=np.array([r["site_id"] for r in rows]))
        manifest.append({"patient_id": pid, "file": filename, "time": duration / 30.4375,
                         "event": int(event), "sites": len(rows)})
        print(f"Prepared {pid}: {len(rows)} sites", flush=True)
    save_json(output / "manifest.json", {"format": "ammr_normalized_crops_v1", "patients": manifest})


class Cohort:
    def __init__(self, manifest):
        manifest = Path(manifest).resolve()
        data = json.loads(manifest.read_text())
        if data["format"] != "ammr_normalized_crops_v1":
            raise ValueError("Unexpected crop format")
        self.root = manifest.parent
        self.rows = data["patients"]
        ids = [row["patient_id"] for row in self.rows]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate patients")
        for row in self.rows:
            if not np.isfinite(row["time"]) or row["time"] <= 0 or row["event"] not in (0, 1):
                raise ValueError("Invalid survival outcome")
            if not (self.root / row["file"]).is_file():
                raise FileNotFoundError(row["file"])

    def metadata(self, index):
        with np.load(self.root / self.rows[index]["file"], allow_pickle=False) as data:
            return data["timing"].astype(float), data["targets"].astype(float), data["event"].copy()

    def fit_transforms(self, indices):
        metadata = [self.metadata(i) for i in indices]
        timing = np.stack([item[0] for item in metadata])
        target_arrays = [np.where((events == 0)[:, None], targets, np.nan) for _, targets, events in metadata]
        targets = np.concatenate(target_arrays)

        def statistics(array):
            means, scales, medians = [], [], []
            for column in array.T:
                valid = column[np.isfinite(column)]
                means.append(float(valid.mean()) if len(valid) else 0.0)
                scales.append(max(float(valid.std()), 1e-8) if len(valid) else 1.0)
                medians.append(float(np.median(valid)) if len(valid) else 0.0)
            return {"mean": means, "scale": scales, "median": medians}

        return {"timing": statistics(timing), "targets": statistics(targets),
                "target_enabled": np.isfinite(targets).any(0).tolist()}

    def load(self, index, transforms, device):
        row = self.rows[index]
        with np.load(self.root / row["file"], allow_pickle=False) as data:
            arrays = {key: data[key].copy() for key in ("pet", "ct", "event", "anatomy", "timing", "targets")}
        n = len(arrays["event"])
        if n < 1 or arrays["pet"].shape != (n, 2, 1, 64, 64, 64) or arrays["ct"].shape != (n, 2, 1, 96, 96, 96):
            raise ValueError(f"Invalid crop shapes: {row['patient_id']}")
        if arrays["targets"].shape != (n, 4) or arrays["timing"].shape != (2,):
            raise ValueError("Invalid target/timing shape")
        if not np.isin(arrays["event"], [0, 1, 2]).all() or not np.isin(arrays["anatomy"], [0, 1, 2, 3]).all():
            raise ValueError("Invalid category IDs")
        for key, low in (("pet", 0), ("ct", -1)):
            if not np.isfinite(arrays[key]).all() or arrays[key].min() < low - 1e-5 or arrays[key].max() > 1 + 1e-5:
                raise ValueError(f"Invalid normalized {key} values")
        missing = ~np.isfinite(arrays["timing"])
        stats = transforms["timing"]
        timing = np.where(missing, stats["median"], arrays["timing"])
        arrays["timing"] = np.r_[(timing - stats["mean"]) / stats["scale"], missing.astype(float)].astype(np.float32)
        valid = np.isfinite(arrays["targets"]) & np.array(transforms["target_enabled"])[None] & (arrays["event"] == 0)[:, None]
        stats = transforms["targets"]
        arrays["targets"] = np.where(valid, (arrays["targets"] - stats["mean"]) / stats["scale"], 0).astype(np.float32)
        arrays["target_valid"] = valid
        result = {key: torch.from_numpy(value).to(device) for key, value in arrays.items()}
        result.update({"time": row["time"], "outcome": row["event"]})
        return result
