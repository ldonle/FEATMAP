from __future__ import annotations

import numpy as np
import torch

from .helpers import (
    ArrayLike,
    check_field_strength_harmonization,
    check_foundation_model_harmonization,
    check_scanner_harmonization,
    load_transformation_matrix,
    prepare_apply_inputs,
    to_numpy_float64,
)


def train_FEATMAP(
    source_embeddings: ArrayLike,
    target_embeddings: ArrayLike,
) -> torch.Tensor:
    """
    Train an affine FEATMAP transformation from paired embeddings.

    Row i in source_embeddings must describe the same observation as row i
    in target_embeddings. Source and target embedding dimensions may differ.
    """
    source = to_numpy_float64(source_embeddings, "source_embeddings")
    target = to_numpy_float64(target_embeddings, "target_embeddings")

    if source.shape[0] != target.shape[0]:
        raise ValueError(
            "source_embeddings and target_embeddings must contain the same "
            f"number of paired observations, but got {source.shape[0]} and "
            f"{target.shape[0]}."
        )

    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)

    source_centered = source - source_mean
    target_centered = target - target_mean

    weights, *_ = np.linalg.lstsq(source_centered, target_centered, rcond=None)
    bias = target_mean - source_mean @ weights

    transformation_matrix = np.vstack([weights, bias[None, :]])
    return torch.from_numpy(transformation_matrix)


def apply_FEATMAP(
    embeddings: ArrayLike,
    transformation_matrix: ArrayLike,
) -> ArrayLike:
    """
    Apply a FEATMAP transformation.

    The output type matches the embeddings input type: NumPy in gives NumPy
    out, and PyTorch in gives PyTorch out.
    """
    embeddings_torch, matrix_torch, input_is_numpy = prepare_apply_inputs(
        embeddings,
        transformation_matrix,
    )

    ones = torch.ones(
        (embeddings_torch.shape[0], 1),
        dtype=matrix_torch.dtype,
        device=embeddings_torch.device,
    )
    transformed = torch.cat([embeddings_torch, ones], dim=1) @ matrix_torch

    if input_is_numpy:
        return transformed.detach().cpu().numpy()
    return transformed


def harmonize_scanner(
    embeddings: ArrayLike,
    source_scanner: str,
    target_scanner: str,
    foundation_model: str,
) -> ArrayLike:
    """Harmonize pathology embeddings between scanners."""
    check_scanner_harmonization(
        source_scanner,
        target_scanner,
        foundation_model,
    )

    filename = f"scanner_{foundation_model}_{source_scanner}_to_{target_scanner}.pt"
    transformation_matrix = load_transformation_matrix(filename)
    return apply_FEATMAP(embeddings, transformation_matrix)


def harmonize_foundation_model(
    embeddings: ArrayLike,
    source_model: str,
    target_model: str,
    scanner: str,
) -> ArrayLike:
    """Harmonize pathology embeddings between foundation models."""
    check_foundation_model_harmonization(
        source_model,
        target_model,
        scanner,
    )

    filename = f"foundation_model_{scanner}_{source_model}_to_{target_model}.pt"
    transformation_matrix = load_transformation_matrix(filename)
    return apply_FEATMAP(embeddings, transformation_matrix)


def harmonize_field_strength(
    embeddings: ArrayLike,
    source_field_strength: str,
    target_field_strength: str,
) -> ArrayLike:
    """Harmonize MRI embeddings between field strengths."""
    check_field_strength_harmonization(
        source_field_strength,
        target_field_strength,
    )

    filename = f"field_strength_{source_field_strength}_to_{target_field_strength}.pt"
    transformation_matrix = load_transformation_matrix(filename)
    return apply_FEATMAP(embeddings, transformation_matrix)
