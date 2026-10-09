"""Generate a PET latent from pure noise with only CT as the condition."""
import torch
from ..models.diffusion import CTControlDiffusion, NoiseScheduler


@torch.inference_mode()
def sample_pet(model: CTControlDiffusion, scheduler: NoiseScheduler,
               ct_control: torch.Tensor, seed: int = 42) -> torch.Tensor:
    if ct_control.ndim != 5 or ct_control.shape[1] != 8 or not torch.isfinite(ct_control).all():
        raise ValueError("CT control must be finite [B,8,D,H,W]")
    was_training = model.training
    model.eval()
    generator = torch.Generator(device=ct_control.device).manual_seed(seed)
    latent = torch.randn(ct_control.shape, generator=generator, device=ct_control.device)
    try:
        for step in reversed(range(scheduler.num_timesteps)):
            timesteps = torch.full((ct_control.shape[0],), step, device=ct_control.device, dtype=torch.long)
            prediction = model(latent, timesteps, ct_control)
            noise = (torch.randn(latent.shape, generator=generator, device=latent.device)
                     if step > 0 else torch.zeros_like(latent))
            latent = scheduler.step(latent, prediction, step, noise)
        if not torch.isfinite(latent).all():
            raise ValueError("Diffusion sampling produced NaN or Inf")
        return latent
    finally:
        model.train(was_training)
