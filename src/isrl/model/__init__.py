from pathlib import Path

import numpy as np

DATA_ = Path("./data")
SANDBOX = DATA_.joinpath("sandbox")
PAPER = DATA_.joinpath("paper")

ArrayLike = np.ndarray | list | (float | int)

def gaussian_kernel(x: np.ndarray, centers: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    centers = np.asarray(centers, dtype=float)
    sigma = np.asarray(sigma, dtype=float).reshape(-1, 1)
    if centers.ndim != 2:
        raise ValueError("centers must be 2D: (h_dim, n_h_cues).")
    if x.shape[0] != centers.shape[0] or sigma.shape[0] != centers.shape[0]:
        raise ValueError("Shape mismatch among h, centers, sigma (h_dim).")
    k = np.exp(-((x - centers) ** 2) / (2.0 * (sigma ** 2)))
    return k / k.sum(axis=1, keepdims=True)
