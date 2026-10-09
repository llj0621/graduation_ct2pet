"""Versioned transforms between PVAE and diffusion latent coordinates."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

__all__ = (
    "LatentArtifactManifest",
    "LatentTransform",
    "assert_matching_contracts",
    "checkpoint_identity",
    "legacy_manifest",
)


LATENT_TRANSFORM_FORMAT_VERSION = 1
LATENT_MANIFEST_FORMAT_VERSION = 1
LATENT_MANIFEST_FILENAME = "latent_manifest.json"
TRANSFORM_TYPES = frozenset(
    {"codebook_minmax", "global_standardized", "per_channel_standardized"}
)
_MIN_SCALE = 1e-8


def _checkpoint_key(identity: Mapping[str, Any]) -> tuple[Any, ...]:
    if identity.get("sha256"):
        return ("sha256", identity["sha256"])
    return (
        "path_size",
        os.path.abspath(str(identity.get("path", ""))),
        identity.get("size_bytes"),
    )


def checkpoint_identity(path: str, *, calculate_sha256: bool = True) -> dict[str, Any]:
    """Return a stable checkpoint identity suitable for artifact contracts."""
    resolved = os.path.abspath(path)
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"PVAE checkpoint not found: {resolved}")
    result: dict[str, Any] = {
        "path": resolved,
        "size_bytes": os.path.getsize(resolved),
    }
    if calculate_sha256:
        digest = hashlib.sha256()
        with open(resolved, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        result["sha256"] = digest.hexdigest()
    return result


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _number_list(value: Any, name: str, channels: int) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be an array")
    result = tuple(
        _finite_number(item, f"{name}[{index}]") for index, item in enumerate(value)
    )
    if len(result) != channels:
        raise ValueError(f"{name} has {len(result)} values; expected {channels}")
    return result


def _save_json_atomic(payload: Mapping[str, Any], path: str | os.PathLike[str]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


@dataclass(frozen=True)
class LatentTransform:
    """Invertible affine mapping from continuous PVAE to diffusion coordinates."""

    transform_type: str
    channels: int
    codebook_min: float
    codebook_max: float
    pvae_checkpoint: Mapping[str, Any]
    quantizer: Mapping[str, Any]
    center: float | tuple[float, ...] | None = None
    scale: float | tuple[float, ...] | None = None
    format_version: int = LATENT_TRANSFORM_FORMAT_VERSION

    def __post_init__(self) -> None:
        if self.format_version != LATENT_TRANSFORM_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported latent-transform format_version={self.format_version}; "
                f"expected {LATENT_TRANSFORM_FORMAT_VERSION}"
            )
        if self.transform_type not in TRANSFORM_TYPES:
            raise ValueError(f"Unsupported latent transform: {self.transform_type}")
        if (
            isinstance(self.channels, bool)
            or not isinstance(self.channels, int)
            or self.channels < 1
        ):
            raise ValueError("channels must be a positive integer")
        code_min = _finite_number(self.codebook_min, "codebook_min")
        code_max = _finite_number(self.codebook_max, "codebook_max")
        if code_max - code_min <= _MIN_SCALE:
            raise ValueError("codebook_max must exceed codebook_min by more than 1e-8")
        if not isinstance(self.pvae_checkpoint, Mapping) or not self.pvae_checkpoint:
            raise ValueError("pvae_checkpoint identity must be a non-empty object")
        if not isinstance(self.quantizer, Mapping) or not self.quantizer:
            raise ValueError("quantizer must be a non-empty object")

        if self.transform_type == "codebook_minmax":
            if self.center is not None or self.scale is not None:
                raise ValueError("codebook_minmax must not define center or scale")
        elif self.transform_type == "global_standardized":
            center = _finite_number(self.center, "center")
            scale = _finite_number(self.scale, "scale")
            if scale <= _MIN_SCALE:
                raise ValueError("scale must be greater than 1e-8")
            object.__setattr__(self, "center", center)
            object.__setattr__(self, "scale", scale)
        else:
            center = _number_list(self.center, "center", self.channels)
            scale = _number_list(self.scale, "scale", self.channels)
            if any(item <= _MIN_SCALE for item in scale):
                raise ValueError("every per-channel scale must be greater than 1e-8")
            object.__setattr__(self, "center", center)
            object.__setattr__(self, "scale", scale)

        object.__setattr__(self, "codebook_min", code_min)
        object.__setattr__(self, "codebook_max", code_max)
        object.__setattr__(self, "pvae_checkpoint", dict(self.pvae_checkpoint))
        object.__setattr__(self, "quantizer", dict(self.quantizer))

    @property
    def identity(self) -> str:
        metadata = self.to_dict()
        metadata["pvae_checkpoint"] = list(_checkpoint_key(self.pvae_checkpoint))
        payload = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _validate_tensor(self, latent: Tensor) -> None:
        if not torch.is_tensor(latent):
            raise TypeError("latent must be a torch.Tensor")
        if latent.ndim not in (4, 5):
            raise ValueError(
                f"Expected [C,D,H,W] or [B,C,D,H,W], got {tuple(latent.shape)}"
            )
        channel_axis = 0 if latent.ndim == 4 else 1
        if latent.shape[channel_axis] != self.channels:
            raise ValueError(
                f"Latent has {latent.shape[channel_axis]} channels; expected {self.channels}"
            )

    def _affine_values(self, latent: Tensor) -> tuple[Tensor | float, Tensor | float]:
        if self.transform_type == "codebook_minmax":
            return 0.0, 1.0
        if self.transform_type == "global_standardized":
            return float(self.center), float(self.scale)
        shape = (
            (self.channels, 1, 1, 1)
            if latent.ndim == 4
            else (1, self.channels, 1, 1, 1)
        )
        center = torch.as_tensor(
            self.center, dtype=latent.dtype, device=latent.device
        ).view(shape)
        scale = torch.as_tensor(
            self.scale, dtype=latent.dtype, device=latent.device
        ).view(shape)
        return center, scale

    def forward(self, pvae_latent: Tensor) -> Tensor:
        self._validate_tensor(pvae_latent)
        base = (
            (pvae_latent - self.codebook_min) / (self.codebook_max - self.codebook_min)
        ) * 2.0 - 1.0
        center, scale = self._affine_values(base)
        return (base - center) / scale

    def inverse(self, diffusion_latent: Tensor) -> Tensor:
        self._validate_tensor(diffusion_latent)
        center, scale = self._affine_values(diffusion_latent)
        base = diffusion_latent * scale + center
        return ((base + 1.0) * 0.5) * (
            self.codebook_max - self.codebook_min
        ) + self.codebook_min

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "format_version": self.format_version,
            "type": self.transform_type,
            "channels": self.channels,
            "pvae_checkpoint": dict(self.pvae_checkpoint),
            "quantizer": dict(self.quantizer),
            "codebook": {"minimum": self.codebook_min, "maximum": self.codebook_max},
        }
        if self.center is not None:
            result["center"] = (
                list(self.center) if isinstance(self.center, tuple) else self.center
            )
            result["scale"] = (
                list(self.scale) if isinstance(self.scale, tuple) else self.scale
            )
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LatentTransform":
        if not isinstance(payload, Mapping):
            raise ValueError("Latent-transform metadata must be an object")
        required = {
            "format_version",
            "type",
            "channels",
            "pvae_checkpoint",
            "quantizer",
            "codebook",
        }
        missing = sorted(required - payload.keys())
        if missing:
            raise ValueError(f"Latent-transform metadata is missing: {missing}")
        codebook = payload["codebook"]
        if (
            not isinstance(codebook, Mapping)
            or "minimum" not in codebook
            or "maximum" not in codebook
        ):
            raise ValueError("codebook must contain minimum and maximum")
        return cls(
            format_version=payload["format_version"],
            transform_type=payload["type"],
            channels=payload["channels"],
            codebook_min=codebook["minimum"],
            codebook_max=codebook["maximum"],
            pvae_checkpoint=payload["pvae_checkpoint"],
            quantizer=payload["quantizer"],
            center=payload.get("center"),
            scale=payload.get("scale"),
        )

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "LatentTransform":
        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def save(self, path: str | os.PathLike[str]) -> None:
        _save_json_atomic(self.to_dict(), path)

    @classmethod
    def legacy_from_embeddings(
        cls,
        embeddings: Tensor,
        *,
        channels: int,
        pvae_checkpoint: Mapping[str, Any],
        quantizer: Mapping[str, Any],
    ) -> "LatentTransform":
        return cls(
            transform_type="codebook_minmax",
            channels=channels,
            codebook_min=float(embeddings.min().item()),
            codebook_max=float(embeddings.max().item()),
            pvae_checkpoint=pvae_checkpoint,
            quantizer=quantizer,
        )


@dataclass(frozen=True)
class LatentArtifactManifest:
    transform: LatentTransform
    storage: Mapping[str, Any]
    format_version: int = LATENT_MANIFEST_FORMAT_VERSION

    def __post_init__(self) -> None:
        if self.format_version != LATENT_MANIFEST_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported latent manifest format_version={self.format_version}"
            )
        if not isinstance(self.storage, Mapping) or "layout" not in self.storage:
            raise ValueError("storage must contain a layout")

    @property
    def channels(self) -> int:
        return self.transform.channels

    @property
    def pvae_checkpoint(self) -> Mapping[str, Any]:
        return self.transform.pvae_checkpoint

    @property
    def quantizer(self) -> Mapping[str, Any]:
        return self.transform.quantizer

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "latent_transform": self.transform.to_dict(),
            "storage": dict(self.storage),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LatentArtifactManifest":
        required = {"format_version", "latent_transform", "storage"}
        missing = sorted(required - payload.keys())
        if missing:
            raise ValueError(f"Latent manifest is missing: {missing}")
        return cls(
            format_version=payload["format_version"],
            transform=LatentTransform.from_dict(payload["latent_transform"]),
            storage=payload["storage"],
        )

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "LatentArtifactManifest":
        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def save(self, path: str | os.PathLike[str]) -> None:
        _save_json_atomic(self.to_dict(), path)

    def latent_path(self, *, basename: str, scan_path: str | None = None) -> str:
        layout = self.storage["layout"]
        root = self.storage.get("root")
        if layout == "flat_by_json_basename":
            if not root:
                raise ValueError("flat_by_json_basename storage requires root")
            return os.path.join(str(root), f"{basename}.pt")
        if layout in {"adjacent_to_scan", "legacy_adjacent_to_scan"}:
            if scan_path is None:
                raise ValueError("adjacent_to_scan requires scan_path")
            from .paths import construct_latent_path

            return construct_latent_path(scan_path)
        raise ValueError(f"Unsupported latent storage layout: {layout}")


def legacy_manifest(
    transform: LatentTransform,
) -> LatentArtifactManifest:
    if transform.transform_type != "codebook_minmax":
        raise ValueError("A legacy manifest requires codebook_minmax")
    return LatentArtifactManifest(
        transform=transform,
        storage={"layout": "adjacent_to_scan"},
    )


def assert_matching_contracts(
    dataset: LatentArtifactManifest,
    checkpoint: LatentArtifactManifest,
) -> None:
    differences = []
    for name, left, right in (
        (
            "transform identity",
            dataset.transform.identity,
            checkpoint.transform.identity,
        ),
        (
            "PVAE checkpoint",
            _checkpoint_key(dataset.pvae_checkpoint),
            _checkpoint_key(checkpoint.pvae_checkpoint),
        ),
        ("quantizer", dict(dataset.quantizer), dict(checkpoint.quantizer)),
    ):
        if left != right:
            differences.append(name)
    if differences:
        raise ValueError(
            "Dataset/checkpoint latent contract mismatch: " + ", ".join(differences)
        )
