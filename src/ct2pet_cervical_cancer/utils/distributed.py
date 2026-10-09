"""torchrun process setup, coordinated main-process work and per-rank RNG state."""
import os
import random
from dataclasses import dataclass
from datetime import timedelta
import numpy as np
import torch
import torch.distributed as dist


@dataclass
class DistributedContext:
    device: torch.device
    rank: int = 0
    world_size: int = 1
    backend: str | None = None

    @property
    def enabled(self):
        return self.world_size > 1

    @property
    def is_main(self):
        return self.rank == 0

    @classmethod
    def from_environment(cls, device_name: str, backend: str = "auto"):
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError("Invalid torchrun RANK/WORLD_SIZE")
        if world_size > 1 and any(name not in os.environ for name in ("RANK", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT")):
            raise ValueError("DDP requires torchrun's RANK, LOCAL_RANK, MASTER_ADDR and MASTER_PORT environment")
        device = torch.device(device_name)
        if device.type not in {"cpu", "cuda"}:
            raise ValueError("CT2PET training supports CPU or CUDA")
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable; select --device cpu")
            if world_size > 1 and device.index is not None:
                raise ValueError("For DDP use --device cuda and select GPUs with CUDA_VISIBLE_DEVICES")
            index = local_rank if world_size > 1 else (device.index or 0)
            if not 0 <= index < torch.cuda.device_count():
                raise ValueError("LOCAL_RANK exceeds visible CUDA devices; launch one process per GPU")
            device = torch.device("cuda", index)
            torch.cuda.set_device(device)
        if world_size == 1:
            return cls(device)
        if not dist.is_available():
            raise RuntimeError("This PyTorch installation does not support distributed training")
        if backend == "auto":
            backend = "nccl" if device.type == "cuda" and dist.is_nccl_available() else "gloo"
        if backend not in {"nccl", "gloo"}:
            raise ValueError("Distributed backend must be auto, nccl or gloo")
        if backend == "nccl" and (device.type != "cuda" or not dist.is_nccl_available()):
            raise ValueError("NCCL requires CUDA and an NCCL-enabled PyTorch build")
        if backend == "gloo" and not dist.is_gloo_available():
            raise ValueError("Gloo is unavailable in this PyTorch build")
        try:
            dist.init_process_group(backend=backend, init_method="env://", rank=rank,
                                    world_size=world_size, timeout=timedelta(hours=2))
        except RuntimeError as error:
            if "unsupported gloo device" in str(error):
                raise RuntimeError("This PyTorch build has no usable Gloo transport. Use a Linux/WSL CUDA "
                                   "environment with NCCL for multi-GPU training, or a build with working Gloo.") from error
            raise
        return cls(device, rank, world_size, backend)

    def on_main(self, operation):
        """Run a filesystem/logging operation once; propagate its result or error."""
        if not self.enabled:
            return operation()
        message = [None]
        if self.is_main:
            try:
                message[0] = {"result": operation(), "error": None}
            except Exception as error:
                message[0] = {"result": None, "error": f"{type(error).__name__}: {error}"}
        dist.broadcast_object_list(message, src=0)
        if message[0]["error"] is not None:
            raise RuntimeError(f"Main process failed: {message[0]['error']}")
        return message[0]["result"]

    def sum(self, values):
        # Gloo control/metric collectives use CPU; NCCL requires CUDA tensors.
        collective_device = self.device if self.backend == "nccl" else torch.device("cpu")
        result = torch.tensor(values, dtype=torch.float64, device=collective_device)
        if self.enabled:
            dist.all_reduce(result, op=dist.ReduceOp.SUM)
        return result.cpu().tolist()

    def rng_states(self):
        local = {"torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
                 "numpy_rng": np.random.get_state(),
                 "cuda_rng": torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else None}
        if not self.enabled:
            return [local]
        states = [None] * self.world_size
        dist.all_gather_object(states, local)
        return states

    def restore_rng(self, checkpoint):
        if "rng_states" in checkpoint:
            states = checkpoint["rng_states"]
            if len(states) != self.world_size:
                raise ValueError("Resume requires the original world size and per-rank RNG states")
            local = states[self.rank]
        else:
            if self.enabled:
                raise ValueError("Legacy single-process checkpoint cannot resume as DDP")
            local = dict(checkpoint)
            cuda = local.get("cuda_rng", [])
            local["cuda_rng"] = cuda[self.device.index] if self.device.type == "cuda" and cuda else None
        torch.set_rng_state(local["torch_rng"])
        random.setstate(local["python_rng"])
        np.random.set_state(local["numpy_rng"])
        if self.device.type == "cuda":
            if local["cuda_rng"] is None:
                raise ValueError("Checkpoint has no CUDA RNG state; exact CPU-to-CUDA resume is unsupported")
            torch.cuda.set_rng_state(local["cuda_rng"], self.device)

    def close(self):
        if self.enabled and dist.is_initialized():
            dist.destroy_process_group()
