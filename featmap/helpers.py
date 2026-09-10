from __future__ import annotations

import json
from pathlib import Path
from typing import Union

import numpy as np
import torch


ArrayLike = Union[np.ndarray, torch.Tensor]

_MANIFEST_PATH = Path(__file__).with_name("manifest.json")
_TRANSFORMATION_MATRIX_DIR = Path(__file__).with_name("transformation_matrices")


def to_numpy_float64(array: ArrayLike, name: str) -> np.ndarray:
    """Convert embeddings to a 2D NumPy float64 array."""
    if isinstance(array, torch.Tensor):
        array = array.detach().cpu().numpy()
    elif not isinstance(array, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array or PyTorch tensor.")

    array = np.asarray(array, dtype=np.float64)

    if array.ndim != 2:
        raise ValueError(f"{name} must be 2D, but got shape {array.shape}.")
    if array.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one embedding.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinite values.")

    return array


def prepare_apply_inputs(
    embeddings: ArrayLike,
    transformation_matrix: ArrayLike,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """Check and prepare inputs for apply_FEATMAP."""
    input_is_numpy = isinstance(embeddings, np.ndarray)

    if input_is_numpy:
        embeddings_torch = torch.from_numpy(np.asarray(embeddings))
    elif isinstance(embeddings, torch.Tensor):
        embeddings_torch = embeddings
    else:
        raise TypeError("embeddings must be a NumPy array or PyTorch tensor.")

    if isinstance(transformation_matrix, np.ndarray):
        matrix_torch = torch.from_numpy(np.asarray(transformation_matrix))
    elif isinstance(transformation_matrix, torch.Tensor):
        matrix_torch = transformation_matrix
    else:
        raise TypeError(
            "transformation_matrix must be a NumPy array or PyTorch tensor."
        )

    if embeddings_torch.ndim != 2:
        raise ValueError(
            "embeddings must be 2D, but got shape "
            f"{tuple(embeddings_torch.shape)}."
        )
    if matrix_torch.ndim != 2:
        raise ValueError(
            "transformation_matrix must be 2D, but got shape "
            f"{tuple(matrix_torch.shape)}."
        )
    if embeddings_torch.shape[1] + 1 != matrix_torch.shape[0]:
        raise ValueError(
            "Embedding dimension does not match the transformation matrix: "
            f"embeddings have dimension {embeddings_torch.shape[1]}, while the "
            f"matrix expects {matrix_torch.shape[0] - 1}."
        )

    if not torch.is_floating_point(matrix_torch):
        matrix_torch = matrix_torch.to(torch.float64)

    matrix_torch = matrix_torch.to(device=embeddings_torch.device)
    embeddings_torch = embeddings_torch.to(dtype=matrix_torch.dtype)

    return embeddings_torch, matrix_torch, input_is_numpy


def _check_name(value: str, name: str, allowed: tuple[str, ...]) -> None:
    if value not in allowed:
        raise ValueError(
            f"Unknown {name} {value!r}. Choose from: {', '.join(allowed)}."
        )


def check_scanner_harmonization(
    source_scanner: str,
    target_scanner: str,
    foundation_model: str,
) -> None:
    """Check names used for scanner harmonization."""
    scanners = ("at2", "gt450", "vs200", "ocus40")
    foundation_models = ("conch", "uni", "virchow2", "hoptimus0", "gigapath")

    _check_name(source_scanner, "source_scanner", scanners)
    _check_name(target_scanner, "target_scanner", scanners)
    _check_name(foundation_model, "foundation_model", foundation_models)

    if source_scanner == target_scanner:
        raise ValueError("source_scanner and target_scanner must be different.")


def check_foundation_model_harmonization(
    source_model: str,
    target_model: str,
    scanner: str,
) -> None:
    """Check names used for foundation-model harmonization."""
    foundation_models = ("conch", "uni", "virchow2", "hoptimus0", "gigapath")
    scanners = ("at2", "gt450", "vs200", "ocus40")

    _check_name(source_model, "source_model", foundation_models)
    _check_name(target_model, "target_model", foundation_models)
    _check_name(scanner, "scanner", scanners)

    if source_model == target_model:
        raise ValueError("source_model and target_model must be different.")


def check_field_strength_harmonization(
    source_field_strength: str,
    target_field_strength: str,
) -> None:
    """Check names used for MRI field-strength harmonization."""
    field_strengths = ("1p5t", "3t")

    _check_name(source_field_strength, "source_field_strength", field_strengths)
    _check_name(target_field_strength, "target_field_strength", field_strengths)

    if source_field_strength == target_field_strength:
        raise ValueError(
            "source_field_strength and target_field_strength must be different."
        )


def load_transformation_matrix(filename: str) -> torch.Tensor:
    """Load and check a FEATMAP transformation matrix."""
    with _MANIFEST_PATH.open("r", encoding="utf-8") as file:
        manifest = json.load(file)

    entry = next(
        (item for item in manifest["transformations"] if item["filename"] == filename),
        None,
    )
    if entry is None:
        raise ValueError(f"No FEATMAP transformation matrix found for {filename!r}.")

    transformation_matrix_path = _TRANSFORMATION_MATRIX_DIR / filename
    if not transformation_matrix_path.exists():
        raise FileNotFoundError(
            f"Could not find {filename!r} in {_TRANSFORMATION_MATRIX_DIR}. "
            "Download and unpack the FEATMAP transformation matrices as "
            "described in README.md."
        )

    transformation_matrix = torch.load(transformation_matrix_path, map_location="cpu")

    if not isinstance(transformation_matrix, torch.Tensor):
        raise TypeError(f"{filename!r} does not contain a PyTorch tensor.")
    if transformation_matrix.ndim != 2:
        raise ValueError(f"{filename!r} must contain a 2D transformation matrix.")

    expected_input_dim = entry.get("input_dim")
    expected_output_dim = entry.get("output_dim")

    if expected_input_dim is not None and transformation_matrix.shape[0] != expected_input_dim + 1:
        raise ValueError(
            f"{filename!r} expects input dimension {expected_input_dim}, "
            f"but the stored matrix has input dimension {transformation_matrix.shape[0] - 1}."
        )
    if expected_output_dim is not None and transformation_matrix.shape[1] != expected_output_dim:
        raise ValueError(
            f"{filename!r} expects output dimension {expected_output_dim}, "
            f"but the stored matrix has output dimension {transformation_matrix.shape[1]}."
        )

    return transformation_matrix
