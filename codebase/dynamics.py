# --- built in ---
import os

# --- 3rd party ---
import numpy as np
import torch
from torch import nn

# --- my module ---


def langevin_dynamics(
    score_fn,
    x,
    eps=0.1,
    n_steps=1000,
    vis_callback= None,
    vis_steps = 10
):
    """Langevin dynamics

    Args:
        score_fn (callable): a score function with the following sign
            func(x: torch.Tensor) -> torch.Tensor
        x (torch.Tensor): input samples
        eps (float, optional): noise scale. Defaults to 0.1.
        n_steps (int, optional): number of steps. Defaults to 1000.
    """
    for i in range(n_steps):
        x = x + eps/2. * score_fn(x).detach()
        x = x + torch.randn_like(x) * np.sqrt(eps)

        if i % vis_steps ==0  and vis_callback is not None:
            vis_callback(x,i)
    return x
