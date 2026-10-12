from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .data import TaskBundle

EPS = 1e-12


def inverse_softplus(value: float) -> float:
    return float(math.log(math.expm1(value)))


def build_bounds(bundle: TaskBundle, indices: np.ndarray, q_step: float, min_width: float = 1e-8) -> np.ndarray:
    pieces = [bundle.deaths[int(index)] for index in np.asarray(indices, dtype=int) if bundle.deaths[int(index)].size]
    if not pieces:
        raise ValueError("Training diagrams contain no positive nontrivial H0 deaths")
    deaths = np.concatenate(pieces).astype(np.float64)
    deaths = deaths[np.isfinite(deaths)]
    deaths = deaths[(deaths > 0.0) & (deaths < 1.0 - 1e-12)]
    if deaths.size < 10:
        raise ValueError(f"Only {deaths.size} eligible H0 deaths in training data")
    quantiles = np.arange(float(q_step), 100.0, float(q_step), dtype=np.float64)
    raw = np.concatenate(([0.0], np.percentile(deaths, quantiles), [1.0]))
    raw = np.sort(np.clip(raw[np.isfinite(raw)], 0.0, 1.0))
    unique = [0.0]
    for value in raw:
        value = float(value)
        if value <= 0.0 or value >= 1.0:
            continue
        if value > unique[-1] + float(min_width):
            unique.append(value)
    if 1.0 > unique[-1] + float(min_width):
        unique.append(1.0)
    else:
        unique[-1] = 1.0
    output = np.asarray(unique, dtype=np.float32)
    if len(output) < 3 or not np.all(np.diff(output) > 0):
        raise ValueError(f"Invalid adaptive bounds for q={q_step}: {output}")
    return output


def sigma_from_bounds(bounds: np.ndarray) -> float:
    # Preserve the original implementation's float32 midpoint arithmetic
    # before promoting to float64 for the log-spacing calculation.
    bounds_array = np.asarray(bounds)
    centers = 0.5 * (bounds_array[:-1] + bounds_array[1:])
    centers = np.asarray(centers, dtype=np.float64)
    centers = np.sort(centers[np.isfinite(centers) & (centers > 0.0)])
    if len(centers) < 2:
        return 0.1
    differences = np.diff(np.log(centers + EPS))
    differences = differences[differences > 0]
    return float(max(2.0 * np.median(differences), 1e-4)) if len(differences) else 0.1


def quadrature(bounds: np.ndarray, points: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    nodes, weights = np.polynomial.legendre.leggauss(int(points))
    nodes_t = torch.tensor(nodes, dtype=torch.float32, device=device)
    weights_t = torch.tensor(weights, dtype=torch.float32, device=device)
    a = torch.tensor(bounds[:-1], dtype=torch.float32, device=device)
    b = torch.tensor(bounds[1:], dtype=torch.float32, device=device)
    half = 0.5 * (b - a)
    middle = 0.5 * (a + b)
    return half[:, None] * nodes_t[None, :] + middle[:, None], half[:, None] * weights_t[None, :]


class BinnedDataset(Dataset):
    def __init__(self, bundle: TaskBundle, indices: np.ndarray, bounds: np.ndarray):
        self.bundle = bundle
        self.indices = np.asarray(indices, dtype=int)
        self.bounds = np.asarray(bounds, dtype=np.float32)
        self.n_intervals = len(self.bounds) - 1

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, local_index: int):
        global_index = int(self.indices[int(local_index)])
        deaths = self.bundle.deaths[global_index]
        if deaths.size:
            upper = np.nextafter(self.bounds[-1], self.bounds[0], dtype=np.float32)
            clipped = np.clip(deaths, self.bounds[0], upper).astype(np.float32, copy=False)
            bins = np.searchsorted(self.bounds, clipped, side="right") - 1
            bins = np.clip(bins, 0, self.n_intervals - 1).astype(np.int64, copy=False)
        else:
            clipped = np.zeros(0, dtype=np.float32)
            bins = np.zeros(0, dtype=np.int64)
        return (
            torch.from_numpy(clipped),
            torch.from_numpy(bins),
            torch.tensor(int(self.bundle.labels[global_index]), dtype=torch.long),
            self.bundle.sample_ids[global_index],
            self.bundle.participant_ids[global_index],
            global_index,
        )


def collate(batch):
    deaths, bins, labels, sample_ids, participant_ids, global_indices = zip(*batch)
    batch_size = len(batch)
    maximum = max(max(int(value.shape[0]) for value in deaths), 1)
    deaths_pad = torch.zeros((batch_size, maximum), dtype=torch.float32)
    bins_pad = torch.zeros((batch_size, maximum), dtype=torch.long)
    mask = torch.zeros((batch_size, maximum), dtype=torch.float32)
    for index, (death_values, bin_values) in enumerate(zip(deaths, bins)):
        count = int(death_values.shape[0])
        if count:
            deaths_pad[index, :count] = death_values
            bins_pad[index, :count] = bin_values
            mask[index, :count] = 1.0
    return (
        deaths_pad,
        bins_pad,
        mask,
        torch.stack(labels),
        list(sample_ids),
        list(participant_ids),
        np.asarray(global_indices, dtype=int),
    )


def make_loader(
    bundle: TaskBundle,
    indices: np.ndarray,
    bounds: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int = 0,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        BinnedDataset(bundle, indices, bounds),
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        collate_fn=collate,
        generator=generator,
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=bool(num_workers),
    )


class AlphaPiModel(nn.Module):
    def __init__(
        self,
        n_intervals: int,
        n_classes: int,
        bounds: np.ndarray,
        sigma_log: float,
        quad_points: int,
        point_chunk: int,
        device: torch.device,
    ):
        super().__init__()
        self.n_intervals = int(n_intervals)
        self.n_classes = int(n_classes)
        self.sigma_log = float(sigma_log)
        self.point_chunk = int(point_chunk)
        self.raw_alpha = nn.Parameter(torch.full((self.n_intervals,), inverse_softplus(1.0), dtype=torch.float32))
        self.centers = nn.Parameter(torch.zeros(self.n_classes, self.n_intervals, dtype=torch.float32))
        self.clf_head = nn.Linear(self.n_intervals, self.n_classes)
        x_points, w_points = quadrature(bounds, quad_points, device)
        self.register_buffer("x_points", x_points)
        self.register_buffer("w_points", w_points)

    def alpha_positive(self) -> torch.Tensor:
        return F.softplus(self.raw_alpha) + 1e-8

    def compute_z(self, deaths: torch.Tensor, bins: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size = deaths.shape[0]
        n_intervals, quad_points = self.x_points.shape
        log_deaths = torch.log(deaths + EPS)
        alpha_per_death = self.alpha_positive()[bins] * mask
        x_flat = self.x_points.reshape(-1)
        w_flat = self.w_points.reshape(-1)
        log_x = torch.log(x_flat + EPS)
        output = torch.zeros((batch_size, len(log_x)), dtype=torch.float32, device=deaths.device)
        inverse = 1.0 / (2.0 * self.sigma_log * self.sigma_log)
        for start in range(0, len(log_x), self.point_chunk):
            end = min(len(log_x), start + self.point_chunk)
            difference = log_x[start:end].view(1, 1, -1) - log_deaths.unsqueeze(2)
            kernels = torch.exp(-(difference * difference) * inverse)
            output[:, start:end] = (kernels * alpha_per_death.unsqueeze(2)).sum(dim=1)
        output = output.view(batch_size, n_intervals, quad_points)
        return (output * w_flat.view(n_intervals, quad_points).unsqueeze(0)).sum(dim=2)

    def forward(self, deaths: torch.Tensor, bins: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.compute_z(deaths, bins, mask)
        return z, self.clf_head(z)

    def distances_to_centers(self, z: torch.Tensor) -> torch.Tensor:
        return ((z.unsqueeze(1) - self.centers.unsqueeze(0)) ** 2).sum(dim=2)

    def alpha_parts(self, z: torch.Tensor, labels: torch.Tensor, margin: float) -> dict[str, torch.Tensor]:
        squared = self.distances_to_centers(z)
        own_squared = torch.gather(squared, 1, labels.view(-1, 1)).view(-1)
        within = own_squared.mean()
        differences = self.centers.unsqueeze(0) - self.centers.unsqueeze(1)
        distances = torch.sqrt((differences ** 2).sum(dim=2) + 1e-12)
        masked = distances + torch.eye(self.n_classes, device=z.device) * 1e9
        nearest_other = masked.min(dim=1).values
        radius = F.relu(torch.sqrt(own_squared + 1e-12) - 0.5 * nearest_other[labels]).pow(2).mean()
        i_index, j_index = torch.triu_indices(self.n_classes, self.n_classes, offset=1, device=z.device)
        between = F.relu(float(margin) - distances[i_index, j_index]).pow(4).mean()
        return {"within": within, "radius": radius, "between": between}


def class_weighted_ce(labels: np.ndarray, n_classes: int, device: torch.device) -> nn.CrossEntropyLoss:
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=int(n_classes)).astype(float)
    inverse = 1.0 / np.maximum(counts, 1.0)
    weights = inverse * (n_classes / inverse.sum())
    return nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))


def full_loss(ce: torch.Tensor, parts: dict[str, torch.Tensor], args: Any) -> torch.Tensor:
    return (
        float(args.lambda_ce) * ce
        + float(args.lambda_within) * parts["within"]
        + float(args.lambda_radius) * parts["radius"]
        + float(args.lambda_between) * parts["between"]
    )
