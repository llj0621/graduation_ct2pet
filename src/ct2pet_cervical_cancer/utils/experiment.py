"""W&B logging with local JSONL history and atomic, resumable checkpoints."""
import json
import os
from pathlib import Path
from typing import Any
import torch


def save_checkpoint(payload: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


class ExperimentLogger:
    def __init__(self, output_dir: Path, config: dict, project: str, entity: str | None,
                 mode: str, name: str | None, run_id: str | None = None, resume: bool = False):
        if mode not in {"online", "offline", "disabled"}:
            raise ValueError("wandb_mode must be online, offline or disabled")
        self.output_dir = output_dir
        self.run = None
        if mode != "disabled":
            import wandb
            os.environ.setdefault("WANDB_CACHE_DIR", str(output_dir / "wandb_cache"))
            kwargs = {"project": project, "entity": entity, "name": name, "config": config,
                      "dir": str(output_dir), "mode": mode}
            if mode == "online" and resume and run_id:
                kwargs.update(id=run_id, resume="must")
            self.run = wandb.init(**kwargs)
            self.run.define_metric("global_step")
            self.run.define_metric("train/*", step_metric="global_step")
            self.run.define_metric("val/*", step_metric="global_step")
            self.run.define_metric("test/*", step_metric="global_step")
            self.run.summary["resume_source_run_id"] = run_id if resume else None

    @property
    def run_id(self):
        return self.run.id if self.run is not None else None

    def log(self, values: dict[str, Any], step: int):
        scalars = {key: value for key, value in values.items()
                   if isinstance(value, (str, int, float, bool)) or value is None}
        with (self.output_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(scalars, global_step=step), allow_nan=False) + "\n")
        if self.run is not None:
            self.run.log(dict(values, global_step=step))

    def images(self, images: dict[str, Any], step: int):
        if self.run is not None:
            import wandb
            self.run.log({key: wandb.Image(value) for key, value in images.items()} | {"global_step": step})

    def finish(self, failed: bool = False):
        if self.run is not None:
            self.run.finish(exit_code=1 if failed else 0)
