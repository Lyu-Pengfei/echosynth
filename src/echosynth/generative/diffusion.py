"""Conditional Gaussian diffusion training and DDIM sampling (paper section 2.4)."""

from __future__ import annotations

from itertools import pairwise
from typing import cast

import torch
from torch import nn

from echosynth.generative.unet_cfg import ClassConditionalUnet
from echosynth.utils.logging import get_logger

LOGGER = get_logger(__name__)


def sigmoid_beta_schedule(timesteps: int, start: float = -3, end: float = 3) -> torch.Tensor:
    """Return the normalized sigmoid cumulative-alpha schedule used by the paper."""
    steps = timesteps + 1
    values = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    start_value = torch.tensor(start).sigmoid()
    end_value = torch.tensor(end).sigmoid()
    alphas_cumprod = (-((values * (end - start) + start).sigmoid()) + end_value) / (
        end_value - start_value
    )
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return cast(torch.Tensor, betas.clamp(0, 0.999).float())


def _extract(values: torch.Tensor, times: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    return values.gather(0, times).reshape(times.shape[0], *((1,) * (len(shape) - 1)))


class GaussianDiffusion(nn.Module):
    """Noise-prediction diffusion with conditional DDPM and DDIM paths."""

    def __init__(
        self,
        model: ClassConditionalUnet,
        *,
        image_size: int = 112,
        timesteps: int = 1000,
        sampling_timesteps: int = 500,
        beta_schedule: str = "sigmoid",
        objective: str = "pred_noise",
        channels: int = 1,
    ) -> None:
        super().__init__()
        if beta_schedule != "sigmoid" or objective != "pred_noise":
            raise ValueError("v1 supports beta_schedule='sigmoid' and objective='pred_noise' only")
        if not 1 <= sampling_timesteps <= timesteps:
            raise ValueError("sampling_timesteps must be in [1, timesteps]")
        self.model = model
        self.image_size = image_size
        self.timesteps = timesteps
        self.sampling_timesteps = sampling_timesteps
        self.channels = channels
        self.diffusion_config = {
            "image_size": image_size,
            "timesteps": timesteps,
            "sampling_timesteps": sampling_timesteps,
            "beta_schedule": beta_schedule,
            "objective": objective,
            "channels": channels,
        }
        betas = sigmoid_beta_schedule(timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        previous = nn.functional.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", previous)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1 / alphas))
        posterior_variance = betas * (1 - previous) / (1 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance.clamp(min=1e-20))

    def q_sample(
        self, images: torch.Tensor, times: torch.Tensor, noise: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Sample q(x_t | x_0)."""
        noise = torch.randn_like(images) if noise is None else noise
        return (
            _extract(self.sqrt_alphas_cumprod, times, images.shape) * images
            + _extract(self.sqrt_one_minus_alphas_cumprod, times, images.shape) * noise
        )

    def forward(self, images: torch.Tensor, classes: torch.Tensor) -> torch.Tensor:
        """Return the epsilon-prediction training loss."""
        if images.shape[1:] != (self.channels, self.image_size, self.image_size):
            raise ValueError(
                f"expected images (*, {self.channels}, {self.image_size}, {self.image_size}), "
                f"got {tuple(images.shape)}"
            )
        times = torch.randint(0, self.timesteps, (len(images),), device=images.device)
        noise = torch.randn_like(images)
        noisy = self.q_sample(images, times, noise)
        predicted = self.model(noisy, times, classes)
        return nn.functional.mse_loss(predicted, noise)

    @torch.no_grad()
    def p_sample(
        self, images: torch.Tensor, times: torch.Tensor, classes: torch.Tensor, cond_scale: float
    ) -> torch.Tensor:
        """Take one ancestral DDPM step."""
        predicted_noise = self.model.forward_with_cond_scale(images, times, classes, cond_scale)
        beta = _extract(self.betas, times, images.shape)
        one_minus = _extract(self.sqrt_one_minus_alphas_cumprod, times, images.shape)
        mean = _extract(self.sqrt_recip_alphas, times, images.shape) * (
            images - beta * predicted_noise / one_minus
        )
        variance = _extract(self.posterior_variance, times, images.shape)
        noise = torch.randn_like(images)
        nonzero = (times > 0).float().reshape(len(times), 1, 1, 1)
        return mean + nonzero * torch.sqrt(variance) * noise

    @torch.no_grad()
    def ddim_sample(self, classes: torch.Tensor, cond_scale: float) -> torch.Tensor:
        """Sample deterministically along the configured reduced DDIM time grid."""
        device = self.betas.device
        batch = len(classes)
        images = torch.randn(batch, self.channels, self.image_size, self.image_size, device=device)
        times = torch.linspace(-1, self.timesteps - 1, self.sampling_timesteps + 1).long()
        descending = list(reversed(times.tolist()))
        pairs = list(pairwise(descending))
        for current, following in pairs:
            time_batch = torch.full((batch,), current, device=device, dtype=torch.long)
            predicted_noise = self.model.forward_with_cond_scale(
                images, time_batch, classes, cond_scale
            )
            alpha = self.alphas_cumprod[current]
            predicted_start = (images - torch.sqrt(1 - alpha) * predicted_noise) / torch.sqrt(alpha)
            predicted_start = predicted_start.clamp(-1, 1)
            if following < 0:
                images = predicted_start
                continue
            next_alpha = self.alphas_cumprod[following]
            images = (
                torch.sqrt(next_alpha) * predicted_start
                + torch.sqrt(1 - next_alpha) * predicted_noise
            )
        return images

    @torch.no_grad()
    def sample(self, classes: torch.Tensor, cond_scale: float = 1.0) -> torch.Tensor:
        """Use DDIM when sampling steps are reduced, otherwise ancestral DDPM."""
        if self.sampling_timesteps < self.timesteps:
            LOGGER.info("sampler=DDIM steps=%d", self.sampling_timesteps)
            return self.ddim_sample(classes, cond_scale)
        LOGGER.info("sampler=DDPM steps=%d", self.timesteps)
        images = torch.randn(
            len(classes), self.channels, self.image_size, self.image_size, device=self.betas.device
        )
        for time in reversed(range(self.timesteps)):
            times = torch.full((len(classes),), time, device=images.device, dtype=torch.long)
            images = self.p_sample(images, times, classes, cond_scale)
        return images
