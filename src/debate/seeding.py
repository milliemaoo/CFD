"""Seeding helper shared by every entry point.

`set_seed` fixes the RNG state of `random`, NumPy and PyTorch (all CUDA
devices included). Every script calls it immediately before each generation,
once per agent per round, so a seed reproduces a run even though decoding
uses temperature sampling."""

import random

import numpy as np
import torch


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        # single GPU
        torch.cuda.manual_seed(seed)
        # multi GPU
        torch.cuda.manual_seed_all(seed)

