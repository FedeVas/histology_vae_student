from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RepresentationModelOutput:
    """
    Унифицированный результат Autoencoder и VAE.

    Для Autoencoder:
        embedding содержит детерминированный latent vector;
        z равен embedding;
        mu и log_var равны None.

    Для VAE:
        embedding содержит posterior mean mu;
        z содержит sampled или deterministic latent vector;
        mu и log_var содержат параметры posterior.
    """

    reconstruction: torch.Tensor
    embedding: torch.Tensor
    z: torch.Tensor

    mu: torch.Tensor | None = None
    log_var: torch.Tensor | None = None