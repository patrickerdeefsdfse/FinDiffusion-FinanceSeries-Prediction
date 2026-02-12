"""
FinDiffusion - Model
Transformer-based Denoising Diffusion Probabilistic Model (DDPM)
with Interpretable Seasonal-Trend Decomposition for Financial Time Series Generation.

Key references:
  [1] Diffusion-TS (ICLR 2024) - Interpretable Diffusion for General Time Series Generation
  [2] FIDE (NeurIPS 2024) - Frequency-Inflated Conditional Diffusion Model
  [3] Ho et al. (NeurIPS 2020) - Denoising Diffusion Probabilistic Models
  [4] Song et al. (ICLR 2021) - Denoising Diffusion Implicit Models (DDIM)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


# ======================================================================
#  Diffusion Noise Schedule
# ======================================================================

def linear_beta_schedule(n_steps: int, beta_start: float, beta_end: float) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, n_steps)


def cosine_beta_schedule(n_steps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine schedule from Nichol & Dhariwal (2021)."""
    steps = torch.arange(n_steps + 1, dtype=torch.float64) / n_steps
    alpha_bar = torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
    return torch.clamp(betas, 0.0001, 0.9999).float()


def get_beta_schedule(name: str, n_steps: int,
                      beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    if name == 'linear':
        return linear_beta_schedule(n_steps, beta_start, beta_end)
    elif name == 'cosine':
        return cosine_beta_schedule(n_steps)
    else:
        raise ValueError(f"Unknown beta schedule: {name}")


# ======================================================================
#  Sinusoidal Positional Encoding
# ======================================================================

class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding for sequence position."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


# ======================================================================
#  Diffusion Timestep Embedding
# ======================================================================

class DiffusionStepEmbedding(nn.Module):
    """Embeds scalar diffusion timestep into a vector using sinusoidal encoding + MLP."""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: [B] integer timesteps -> [B, d_model] embeddings."""
        half = self.d_model // 2
        freq = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device).float() / half
        )
        args = t.float().unsqueeze(1) * freq.unsqueeze(0)   # [B, half]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [B, d_model]
        return self.mlp(emb)


# ======================================================================
#  Adaptive Layer Normalization (conditions on diffusion step)
# ======================================================================

class AdaptiveLayerNorm(nn.Module):
    """LayerNorm + affine modulation conditioned on diffusion step embedding."""

    def __init__(self, d_model: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.scale = nn.Linear(d_model, d_model)
        self.shift = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, D]  cond: [B, D]
        """
        x = self.norm(x)
        s = self.scale(cond).unsqueeze(1)  # [B, 1, D]
        b = self.shift(cond).unsqueeze(1)
        return x * (1.0 + s) + b


# ======================================================================
#  Transformer Encoder Block (with diffusion-step conditioning)
# ======================================================================

class CondTransformerBlock(nn.Module):
    """Single Transformer encoder block conditioned on diffusion timestep."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.aln1 = AdaptiveLayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.aln2 = AdaptiveLayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, step_emb: torch.Tensor) -> torch.Tensor:
        # Multi-head self-attention
        h = self.aln1(x, step_emb)
        h, _ = self.attn(h, h, h)
        x = x + h
        # Feed-forward
        h = self.aln2(x, step_emb)
        h = self.ffn(h)
        x = x + h
        return x


# ======================================================================
#  Interpretable Decoder: Trend + Seasonal Decomposition
#  Inspired by Diffusion-TS (ICLR 2024)
# ======================================================================

class TrendDecoder(nn.Module):
    """Extracts smooth trend via moving-average pooling + linear projection."""

    def __init__(self, d_model: int, output_dim: int, kernel_size: int = 5):
        super().__init__()
        pad = kernel_size // 2
        self.avg_pool = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=pad)
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, L, D]
        Returns:
            trend_output: [B, L, output_dim]  projected trend
            trend_hidden: [B, L, D]           hidden-space smoothed representation
        """
        smoothed = self.avg_pool(x.transpose(1, 2)).transpose(1, 2)  # [B, L, D]
        return self.proj(smoothed), smoothed


class SeasonalDecoder(nn.Module):
    """Extracts periodic/seasonal patterns via cross-attention on de-trended residual."""

    def __init__(self, d_model: int, n_heads: int, output_dim: int, dropout: float):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, output_dim),
        )

    def forward(self, residual: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """
        residual: [B, L, D]  de-trended residual (encoder_out - trend_hidden)
        memory:   [B, L, D]  full encoder output
        """
        h, _ = self.cross_attn(residual, memory, memory)
        h = self.norm(residual + h)
        return self.ffn(h)


# ======================================================================
#  Denoising Network  (Encoder-Decoder Transformer)
# ======================================================================

class DenoisingNetwork(nn.Module):
    """
    Transformer-based denoising network that predicts x_0 directly.
    Uses interpretable seasonal-trend decomposition in the decoder.
    """

    def __init__(self, feature_dim: int, d_model: int, n_heads: int,
                 n_layers: int, d_ff: int, dropout: float, trend_kernel: int = 5):
        super().__init__()

        # Input projection
        self.input_proj = nn.Linear(feature_dim, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model)
        self.step_emb = DiffusionStepEmbedding(d_model)

        # Transformer encoder (conditioned on diffusion step)
        self.encoder = nn.ModuleList([
            CondTransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Interpretable decoder
        self.trend_decoder = TrendDecoder(d_model, feature_dim, trend_kernel)
        self.seasonal_decoder = SeasonalDecoder(d_model, n_heads, feature_dim, dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x_t: noisy time series [B, L, C]
            t:   diffusion timestep [B]
        Returns:
            x_0_pred: predicted clean signal  [B, L, C]
            trend:    trend component         [B, L, C]
            seasonal: seasonal component      [B, L, C]
        """
        # Encode
        h = self.dropout(self.pos_enc(self.input_proj(x_t)))
        step = self.step_emb(t)

        for block in self.encoder:
            h = block(h, step)

        # Decode: trend first, then seasonal on de-trended residual
        trend, trend_hidden = self.trend_decoder(h)
        residual = h - trend_hidden              # de-trended residual in hidden space
        seasonal = self.seasonal_decoder(residual, h)

        x_0_pred = trend + seasonal
        return x_0_pred, trend, seasonal


# ======================================================================
#  FinDiffusion:  Complete DDPM Model
# ======================================================================

class FinDiffusion(nn.Module):
    """
    Denoising Diffusion Probabilistic Model for Financial Time Series.

    Training:   predict x_0 directly (signal-prediction parameterization)
    Loss:       MSE(x_0, x_0_hat) + lambda * FFT_loss(x_0, x_0_hat)
    Sampling:   standard DDPM reverse process  or  DDIM (faster)
    """

    def __init__(self, feature_dim: int, d_model: int = 128, n_heads: int = 4,
                 n_layers: int = 4, d_ff: int = 512, dropout: float = 0.1,
                 n_steps: int = 200, beta_start: float = 1e-4,
                 beta_end: float = 0.02, beta_schedule: str = 'linear',
                 trend_kernel: int = 5, clip_range: Optional[float] = 5.0):
        super().__init__()

        self.n_steps = n_steps
        self.feature_dim = feature_dim
        self.clip_range = clip_range

        # Denoising network
        self.net = DenoisingNetwork(
            feature_dim, d_model, n_heads, n_layers, d_ff, dropout, trend_kernel
        )

        # Pre-compute diffusion constants
        betas = get_beta_schedule(beta_schedule, n_steps, beta_start, beta_end)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_prev = F.pad(alpha_bars[:-1], (1, 0), value=1.0)

        # Posterior variance  (for DDPM sampling)
        posterior_var = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('alpha_bars_prev', alpha_bars_prev)
        self.register_buffer('sqrt_alpha_bars', torch.sqrt(alpha_bars))
        self.register_buffer('sqrt_one_minus_alpha_bars', torch.sqrt(1.0 - alpha_bars))
        self.register_buffer('posterior_var', posterior_var)
        self.register_buffer('posterior_log_var', torch.log(posterior_var.clamp(min=1e-20)))

    # ---- Forward diffusion  q(x_t | x_0) ----
    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor,
                 noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_0)
        a = self.sqrt_alpha_bars[t].view(-1, 1, 1)
        b = self.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1)
        return a * x_0 + b * noise

    # ---- Training loss ----
    def _frequency_inflated_loss(self, x_pred: torch.Tensor, x_true: torch.Tensor,
                                 gamma: float = 1.5) -> torch.Tensor:
        """
        FIDE-inspired: emphasize higher frequencies in the spectrum.
        Returns weighted MSE on FFT magnitude.
        """
        fft_pred = torch.fft.rfft(x_pred, dim=1)
        fft_true = torch.fft.rfft(x_true, dim=1)
        freq_len = fft_pred.shape[1]
        freqs = torch.linspace(0.0, 1.0, freq_len, device=x_pred.device)
        weights = freqs.pow(gamma)
        weights[0] = 0.0
        w = weights.view(1, -1, 1)
        diff = torch.abs(fft_pred) - torch.abs(fft_true)
        return torch.mean((diff * w) ** 2)

    def _lowpass(self, x: torch.Tensor, kernel_size: int) -> torch.Tensor:
        """Simple low-pass filter via moving average along time axis."""
        if kernel_size <= 1:
            return x
        pad = kernel_size // 2
        x_t = x.transpose(1, 2)  # [B, C, L]
        x_f = F.avg_pool1d(x_t, kernel_size=kernel_size, stride=1, padding=pad)
        return x_f.transpose(1, 2)

    def compute_loss(self, x_0: torch.Tensor, fft_weight: float = 0.1,
                     fide_weight: float = 0.0, fide_gamma: float = 1.5,
                     fide_scales: Optional[list] = None) -> dict:
        B = x_0.shape[0]
        t = torch.randint(0, self.n_steps, (B,), device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise)

        x_0_pred, trend, seasonal = self.net(x_t, t)

        # Reconstruction loss (MSE)
        mse_loss = F.mse_loss(x_0_pred, x_0)

        # Fourier loss: match both amplitude and phase in frequency domain
        fft_pred = torch.fft.rfft(x_0_pred, dim=1)
        fft_real = torch.fft.rfft(x_0, dim=1)
        fft_amp_loss = F.mse_loss(torch.abs(fft_pred), torch.abs(fft_real))
        fft_phase_loss = F.mse_loss(torch.angle(fft_pred), torch.angle(fft_real))
        fft_loss = fft_amp_loss + fft_phase_loss

        # FIDE-inspired multi-scale frequency inflation
        fide_loss = torch.tensor(0.0, device=x_0.device)
        if fide_weight > 0:
            scales = fide_scales if fide_scales else [1]
            total_fide = 0.0
            for s in scales:
                if s <= 0:
                    continue
                if s == 1:
                    x_pred_s = x_0_pred
                    x_true_s = x_0
                else:
                    kernel = max(3, 2 * s + 1)
                    x_pred_lp = self._lowpass(x_0_pred, kernel)
                    x_true_lp = self._lowpass(x_0, kernel)
                    x_pred_s = x_pred_lp[:, ::s, :]
                    x_true_s = x_true_lp[:, ::s, :]
                total_fide = total_fide + self._frequency_inflated_loss(
                    x_pred_s, x_true_s, gamma=fide_gamma
                )
            fide_loss = total_fide / max(1, len(scales))

        total = mse_loss + fft_weight * fft_loss + fide_weight * fide_loss

        return {
            'total': total,
            'mse': mse_loss.item(),
            'fft': fft_loss.item(),
            'fide': float(fide_loss.item()) if torch.is_tensor(fide_loss) else float(fide_loss),
        }

    # ---- DDPM Sampling ----
    @torch.no_grad()
    def sample_ddpm(self, n_samples: int, seq_len: int,
                    device: torch.device) -> torch.Tensor:
        """Standard DDPM ancestral sampling."""
        self.eval()
        x = torch.randn(n_samples, seq_len, self.feature_dim, device=device)

        for t in reversed(range(self.n_steps)):
            t_batch = torch.full((n_samples,), t, device=device, dtype=torch.long)
            x_0_pred, _, _ = self.net(x, t_batch)
            if self.clip_range is not None:
                x_0_pred = torch.clamp(x_0_pred, -self.clip_range, self.clip_range)

            if t > 0:
                alpha = self.alphas[t]
                alpha_bar = self.alpha_bars[t]
                alpha_bar_prev = self.alpha_bars_prev[t]
                beta = self.betas[t]

                # Posterior mean
                coef1 = (torch.sqrt(alpha_bar_prev) * beta) / (1.0 - alpha_bar)
                coef2 = (torch.sqrt(alpha) * (1.0 - alpha_bar_prev)) / (1.0 - alpha_bar)
                mean = coef1 * x_0_pred + coef2 * x

                # Add noise
                var = self.posterior_var[t]
                z = torch.randn_like(x)
                x = mean + torch.sqrt(var) * z
            else:
                x = x_0_pred

        self.train()
        return x

    # ---- DDIM Sampling (faster) ----
    @torch.no_grad()
    def sample_ddim(self, n_samples: int, seq_len: int,
                    device: torch.device, n_ddim_steps: int = 50,
                    eta: float = 0.0) -> torch.Tensor:
        """
        DDIM sampling (Song et al., ICLR 2021) for faster generation.
        eta=0: deterministic;  eta=1: equivalent to DDPM.
        """
        self.eval()

        # Evenly-spaced sub-sequence of timesteps (robust to non-divisible cases)
        timesteps = torch.linspace(self.n_steps - 1, 0, n_ddim_steps).long().tolist()

        x = torch.randn(n_samples, seq_len, self.feature_dim, device=device)

        for i, t in enumerate(timesteps):
            t_batch = torch.full((n_samples,), t, device=device, dtype=torch.long)
            x_0_pred, _, _ = self.net(x, t_batch)
            if self.clip_range is not None:
                x_0_pred = torch.clamp(x_0_pred, -self.clip_range, self.clip_range)

            if i < len(timesteps) - 1:
                t_next = timesteps[i + 1]
            else:
                t_next = 0

            alpha_bar_t = self.alpha_bars[t]
            alpha_bar_next = self.alpha_bars[t_next] if t_next > 0 else torch.tensor(1.0)

            # Predicted noise
            eps_pred = (x - torch.sqrt(alpha_bar_t) * x_0_pred) / torch.sqrt(1 - alpha_bar_t)

            # DDIM update
            sigma = eta * torch.sqrt(
                (1 - alpha_bar_next) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_next)
            )
            dir_xt = torch.sqrt(1 - alpha_bar_next - sigma ** 2) * eps_pred
            noise = torch.randn_like(x) if sigma > 0 else 0

            x = torch.sqrt(alpha_bar_next) * x_0_pred + dir_xt + sigma * noise

        self.train()
        return x

    # ---- Conditional generation (forecasting / imputation) ----
    @torch.no_grad()
    def conditional_sample(self, x_obs: torch.Tensor, mask: torch.Tensor,
                           device: torch.device) -> torch.Tensor:
        """
        Inpainting-style conditional generation.
        mask=1 for observed positions, mask=0 for positions to generate.
        """
        self.eval()
        B, L, C = x_obs.shape
        x = torch.randn_like(x_obs)

        for t in reversed(range(self.n_steps)):
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)

            # Noise the observation to match current noise level
            x_obs_noised = self.q_sample(x_obs, t_batch)

            # Replace observed positions
            x = mask * x_obs_noised + (1 - mask) * x

            # Predict x_0
            x_0_pred, _, _ = self.net(x, t_batch)
            if self.clip_range is not None:
                x_0_pred = torch.clamp(x_0_pred, -self.clip_range, self.clip_range)

            if t > 0:
                alpha_bar = self.alpha_bars[t]
                alpha_bar_prev = self.alpha_bars_prev[t]
                beta = self.betas[t]
                alpha = self.alphas[t]

                coef1 = (torch.sqrt(alpha_bar_prev) * beta) / (1.0 - alpha_bar)
                coef2 = (torch.sqrt(alpha) * (1.0 - alpha_bar_prev)) / (1.0 - alpha_bar)
                mean = coef1 * x_0_pred + coef2 * x

                var = self.posterior_var[t]
                z = torch.randn_like(x)
                x = mean + torch.sqrt(var) * z
            else:
                x = x_0_pred

            # Re-apply observation mask
            if t > 0:
                x = mask * x_obs_noised + (1 - mask) * x
            else:
                x = mask * x_obs + (1 - mask) * x

        self.train()
        return x

    def forward(self, x_0: torch.Tensor, fft_weight: float = 0.1,
                fide_weight: float = 0.0, fide_gamma: float = 1.5,
                fide_scales: Optional[list] = None) -> dict:
        """Forward pass = compute training loss."""
        return self.compute_loss(x_0, fft_weight, fide_weight, fide_gamma, fide_scales)


# ======================================================================
#  Baseline: LSTM-based Generator  (for comparison)
# ======================================================================

class LSTMGenerator(nn.Module):
    """
    Simple LSTM-based autoregressive generator baseline.
    Trained via teacher-forcing to predict next step, then sampled autoregressively.
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 128,
                 n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.lstm = nn.LSTM(
            feature_dim, hidden_dim, n_layers,
            batch_first=True, dropout=dropout if n_layers > 1 else 0
        )
        self.output_proj = nn.Linear(hidden_dim, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Teacher-forcing: predict x[:,1:,:] from x[:,:-1,:]."""
        out, _ = self.lstm(x[:, :-1, :])
        return self.output_proj(out)

    def compute_loss(self, x: torch.Tensor) -> dict:
        pred = self.forward(x)
        target = x[:, 1:, :]
        loss = F.mse_loss(pred, target)
        return {'total': loss, 'mse': loss.item()}

    @torch.no_grad()
    def sample(self, n_samples: int, seq_len: int, device: torch.device,
               clip_range: Optional[float] = None) -> torch.Tensor:
        """Autoregressive sampling."""
        self.eval()
        # Random initial token
        x = torch.randn(n_samples, 1, self.feature_dim, device=device) * 0.1
        h = None

        generated = [x]
        for _ in range(seq_len - 1):
            out, h = self.lstm(x, h)
            x_next = self.output_proj(out)
            noise = torch.randn_like(x_next) * 0.02
            x = x_next + noise
            if clip_range is not None:
                x = torch.clamp(x, -clip_range, clip_range)
            generated.append(x)

        self.train()
        return torch.cat(generated, dim=1)


# ======================================================================
#  Baseline: VAE-based Generator  (stronger comparison)
# ======================================================================

class VAEGenerator(nn.Module):
    """
    Variational Autoencoder baseline for time series generation.
    LSTM encoder -> latent z -> LSTM decoder.
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 128,
                 latent_dim: int = 32, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.n_layers = n_layers

        # Encoder
        self.encoder = nn.LSTM(
            feature_dim, hidden_dim, n_layers,
            batch_first=True, dropout=dropout if n_layers > 1 else 0
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(
            feature_dim, hidden_dim, n_layers,
            batch_first=True, dropout=dropout if n_layers > 1 else 0
        )
        self.output_proj = nn.Linear(hidden_dim, feature_dim)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _, (h_n, _) = self.encoder(x)
        h = h_n[-1]  # Last layer hidden state [B, hidden_dim]
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, seq_len: int) -> torch.Tensor:
        h = self.latent_to_hidden(z)  # [B, hidden_dim]
        # Initialize decoder hidden state
        h0 = h.unsqueeze(0).repeat(self.n_layers, 1, 1)
        c0 = torch.zeros_like(h0)

        # Teacher-free decoding: feed previous output back
        B = z.shape[0]
        x = torch.zeros(B, 1, self.feature_dim, device=z.device)
        outputs = []
        hidden = (h0, c0)
        for _ in range(seq_len):
            out, hidden = self.decoder(x, hidden)
            x = self.output_proj(out)
            outputs.append(x)
        return torch.cat(outputs, dim=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, x.size(1))
        return recon, mu, logvar

    def compute_loss(self, x: torch.Tensor, kl_weight: float = 0.1) -> dict:
        recon, mu, logvar = self.forward(x)
        recon_loss = F.mse_loss(recon, x)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        total = recon_loss + kl_weight * kl_loss
        return {'total': total, 'mse': recon_loss.item(), 'kl': kl_loss.item()}

    @torch.no_grad()
    def sample(self, n_samples: int, seq_len: int, device: torch.device,
               clip_range: Optional[float] = None) -> torch.Tensor:
        self.eval()
        z = torch.randn(n_samples, self.latent_dim, device=device)
        samples = self.decode(z, seq_len)
        self.train()
        if clip_range is not None:
            samples = torch.clamp(samples, -clip_range, clip_range)
        return samples
