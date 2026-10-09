"""Linear DDPM schedule and Min-SNR weighted v-prediction used by Leaper B1."""
import torch
import torch.nn as nn


class NoiseScheduler(nn.Module):
    def __init__(self, num_timesteps: int = 1000, beta_start: float = 0.0001,
                 beta_end: float = 0.02, min_snr_gamma: float = 5.0):
        super().__init__()
        if num_timesteps < 2 or not 0 < beta_start < beta_end < 1 or min_snr_gamma <= 0:
            raise ValueError("Invalid DDPM timesteps, beta range or Min-SNR gamma")
        self.num_timesteps = num_timesteps
        self.min_snr_gamma = min_snr_gamma
        betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", 1 - betas)
        self.register_buffer("alphas_cumprod", torch.cumprod(1 - betas, dim=0))

    def coefficients(self, timesteps: torch.Tensor):
        alpha_bar = self.alphas_cumprod[timesteps].reshape(-1, 1, 1, 1, 1)
        return alpha_bar.sqrt(), (1 - alpha_bar).sqrt()

    def add_noise(self, clean: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor):
        signal, noise_scale = self.coefficients(timesteps)
        return signal * clean + noise_scale * noise

    def velocity(self, clean: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor):
        signal, noise_scale = self.coefficients(timesteps)
        return signal * noise - noise_scale * clean

    def predict_x0(self, noisy: torch.Tensor, velocity: torch.Tensor, timesteps: torch.Tensor):
        signal, noise_scale = self.coefficients(timesteps)
        return signal * noisy - noise_scale * velocity

    def loss(self, prediction: torch.Tensor, target: torch.Tensor, timesteps: torch.Tensor):
        alpha_bar = self.alphas_cumprod[timesteps]
        snr = alpha_bar / (1 - alpha_bar)
        weights = snr.clamp(max=self.min_snr_gamma) / (snr + 1)
        mse = (prediction.float() - target.float()).square().flatten(1).mean(1)
        return (weights * mse).mean()

    def step(self, noisy: torch.Tensor, velocity: torch.Tensor, timestep: int,
             noise: torch.Tensor) -> torch.Tensor:
        # Match the source B1 sampler's beta-variance DDPM update, without clipping.
        alpha_bar = self.alphas_cumprod[timestep]
        epsilon = (1 - alpha_bar).sqrt() * noisy + alpha_bar.sqrt() * velocity
        mean = (noisy - self.betas[timestep] / (1 - alpha_bar).sqrt() * epsilon) / self.alphas[timestep].sqrt()
        return mean + self.betas[timestep].sqrt() * noise if timestep > 0 else mean
