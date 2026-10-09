"""Paired case-directory CT/PET latents, fixed splits and shape-aware batches."""
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
import torch
from torch.utils.data import Dataset, Sampler
from .latent_transform import LatentTransform, checkpoint_identity


def stable_seed(seed: int, case: str) -> int:
    digest = hashlib.sha256(f"ct2pet:{seed}:{case}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def load_tensor(path: Path) -> torch.Tensor:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Expected a tensor in {path}")
    if value.ndim == 5 and value.shape[0] == 1:
        value = value[0]
    if (value.ndim != 4 or value.shape[0] != 8 or min(value.shape[1:]) < 2
            or not torch.isfinite(value).all()):
        raise ValueError(f"Expected finite [8,D,H,W] latent with spatial size >=2 in {path}")
    return value.float().contiguous()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_transform(checkpoint: Path) -> LatentTransform:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = payload.get("hyper_parameters", {})
    state = payload.get("state_dict", payload)
    embeddings = next((state[key] for key in ("codebook.codebook.embeddings", "codebook.embeddings")
                       if key in state), None)
    if embeddings is None or tuple(embeddings.shape) != (config.get("n_codes", 8192), 8):
        raise ValueError(f"Cannot find an eight-channel VQ codebook in {checkpoint}")
    return LatentTransform.legacy_from_embeddings(
        embeddings, channels=8, pvae_checkpoint=checkpoint_identity(str(checkpoint)),
        quantizer={"n_codes": int(embeddings.shape[0]), "embedding_dim": 8,
                   "assignment_only_l2": bool(config.get("assignment_only_l2", False)),
                   "spherical_vq": bool(config.get("spherical_vq", False))},
    )


def discover_pairs(input_root: Path) -> list[dict]:
    if not input_root.is_dir():
        raise FileNotFoundError(input_root)
    records = []
    for directory in sorted(input_root.iterdir()):
        if not directory.is_dir():
            continue
        ct_path, pet_path = directory / "ct.pt", directory / "pet.pt"
        if not ct_path.is_file() and not pet_path.is_file():
            continue
        if not ct_path.is_file() or not pet_path.is_file():
            raise FileNotFoundError(f"Incomplete CT/PET pair: {directory}")
        ct, pet = load_tensor(ct_path), load_tensor(pet_path)
        if ct.shape != pet.shape:
            raise ValueError(f"CT/PET latent shapes differ in {directory}")
        metadata = directory / "latent_metadata.json"
        if metadata.is_file():
            encoding = json.loads(metadata.read_text(encoding="utf-8-sig")).get("encoding", {})
            if encoding.get("quantize") is True:
                raise ValueError(f"B1 expects continuous encoder latents, but {metadata} declares quantize=true")
        records.append({"case": directory.name, "ct_latent": str(ct_path.resolve()),
                        "pet_latent": str(pet_path.resolve()), "shape": list(ct.shape),
                        "ct_sha256": file_sha256(ct_path), "pet_sha256": file_sha256(pet_path)})
    if not records:
        raise ValueError(f"No case directories containing ct.pt/pet.pt in {input_root}")
    return records


def resolve_split(records: list[dict], path: Path, seed: int, validation_fraction: float,
                  test_fraction: float, patient_map: Path | None = None) -> dict:
    if not 0 < validation_fraction < 1 or not 0 < test_fraction < 1 or validation_fraction + test_fraction >= 1:
        raise ValueError("Validation/test fractions must be positive and sum to less than one")
    by_case = {record["case"]: record for record in records}
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        rows = payload.get("records", [])
        if len(rows) != len(by_case) or {row["case"] for row in rows} != by_case.keys():
            raise ValueError("Split manifest must contain every current case exactly once")
        for row in rows:
            current = by_case[row["case"]]
            if any(key in row and row[key] != current[key]
                   for key in ("shape", "ct_latent", "pet_latent", "ct_sha256", "pet_sha256")):
                raise ValueError(f"Split data changed for {row['case']}")
            row.update(current)
    else:
        mapping = (json.loads(patient_map.read_text(encoding="utf-8-sig")) if patient_map else {})
        if patient_map and (not isinstance(mapping, dict) or by_case.keys() - mapping.keys()):
            raise ValueError("Patient map must be a JSON object mapping every case to a patient ID")
        rows = [dict(record, patient_id=str(mapping.get(record["case"], record["case"]))) for record in records]
        groups = sorted({row["patient_id"] for row in rows})
        if len(groups) < 3:
            raise ValueError("Need at least three patient/case groups for train, validation and test")
        random.Random(seed).shuffle(groups)
        n_test = max(1, round(len(groups) * test_fraction))
        n_val = max(1, round(len(groups) * validation_fraction))
        if n_test + n_val >= len(groups):
            raise ValueError("Requested split leaves no training groups")
        assignments = {group: ("test" if index < n_test else "validation" if index < n_test + n_val else "train")
                       for index, group in enumerate(groups)}
        for row in rows:
            row["split"] = assignments[row["patient_id"]]
        payload = {"format": "ct2pet_paired_latent_split_v1", "seed": seed,
                   "grouping": "patient_id" if patient_map else "case_id", "records": rows}
    seen_groups = {}
    for row in rows:
        if row.get("split") not in {"train", "validation", "test"}:
            raise ValueError(f"Unknown split for {row['case']}")
        group = str(row.get("patient_id", row["case"]))
        if not group.strip() or group in {"None", "null"}:
            raise ValueError("Patient IDs cannot be empty or null")
        if group in seen_groups and seen_groups[group] != row["split"]:
            raise ValueError(f"Patient/case group occurs across splits: {group}")
        seen_groups[group] = row["split"]
        row["patient_id"] = group
    if {row["split"] for row in rows} != {"train", "validation", "test"}:
        raise ValueError("Train, validation and test splits must all be nonempty")
    if patient_map:
        mapping = json.loads(patient_map.read_text(encoding="utf-8-sig"))
        if not isinstance(mapping, dict) or any(str(mapping.get(row["case"])) != row["patient_id"] for row in rows):
            raise ValueError("Patient mapping differs from the resolved split manifest")
    payload["records"] = rows
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


class PairedLatentDataset(Dataset):
    def __init__(self, records: list[dict], ct_transform: LatentTransform,
                 pet_transform: LatentTransform, input_space: str):
        if input_space not in {"raw", "codebook_minmax"} or not records:
            raise ValueError("Expected nonempty records and input_space raw/codebook_minmax")
        self.records = records
        self.ct_transform = ct_transform
        self.pet_transform = pet_transform
        self.input_space = input_space

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        ct, pet = load_tensor(Path(row["ct_latent"])), load_tensor(Path(row["pet_latent"]))
        if ct.shape != pet.shape or list(ct.shape) != row["shape"]:
            raise ValueError(f"Latent shape changed for {row['case']}")
        if self.input_space == "raw":
            ct, pet = self.ct_transform.forward(ct), self.pet_transform.forward(pet)
        return {"case": row["case"], "ct": ct, "pet": pet}


class ShapeBatchSampler(Sampler):
    """Batch identical crop shapes without resizing or padding the latent target."""
    def __init__(self, records: list[dict], batch_size: int, shuffle: bool, seed: int):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.groups = defaultdict(list)
        for index, row in enumerate(records):
            self.groups[tuple(row["shape"])].append(index)
        self.batch_size, self.shuffle, self.seed, self.epoch = batch_size, shuffle, seed, 0

    def __len__(self):
        return sum(math.ceil(len(indices) / self.batch_size) for indices in self.groups.values())

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        batches = []
        for values in self.groups.values():
            indices = list(values)
            if self.shuffle:
                rng.shuffle(indices)
            batches.extend(indices[start:start + self.batch_size] for start in range(0, len(indices), self.batch_size))
        if self.shuffle:
            rng.shuffle(batches)
        yield from batches


class DistributedShapeBatchSampler(ShapeBatchSampler):
    """Equal training steps with shape-homogeneous global batches; exact eval shards."""
    def __init__(self, records, batch_size: int, shuffle: bool, seed: int,
                 rank: int, world_size: int, training: bool):
        super().__init__(records, batch_size, shuffle, seed)
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError("Invalid distributed sampler rank/world_size")
        self.rank, self.world_size, self.training = rank, world_size, training

    def __len__(self):
        if self.training:
            return sum(math.ceil(len(indices) / (self.batch_size * self.world_size))
                       for indices in self.groups.values())
        return sum(math.ceil(len(indices[self.rank::self.world_size]) / self.batch_size)
                   for indices in self.groups.values())

    def __iter__(self):
        if not self.training:
            for indices in self.groups.values():
                local = indices[self.rank::self.world_size]
                for start in range(0, len(local), self.batch_size):
                    yield local[start:start + self.batch_size]
            return
        rng = random.Random(self.seed + self.epoch)
        batches = []
        width = self.batch_size * self.world_size
        for values in self.groups.values():
            indices = list(values)
            if self.shuffle:
                rng.shuffle(indices)
            for start in range(0, len(indices), width):
                batch = indices[start:start + width]
                # Every rank needs >=1 sample for a synchronized backward pass.
                # Repeat only tiny tails; otherwise uneven local sizes are loss-weighted.
                if len(batch) < self.world_size:
                    batch = [batch[index % len(batch)] for index in range(self.world_size)]
                batches.append(batch)
        if self.shuffle:
            rng.shuffle(batches)
        for batch in batches:
            yield batch[self.rank::self.world_size]
