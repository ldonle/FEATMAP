"""Minimal FEATMAP training/application example using synthetic embeddings."""

import sys
from pathlib import Path

import numpy as np

# Allow this file to be run directly from the repository without installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from featmap import apply_FEATMAP, train_FEATMAP


rng = np.random.default_rng(0)

# Create paired synthetic embeddings.
source_embeddings = rng.normal(size=(100, 8))
true_matrix = rng.normal(size=(9, 5))  # 8 input dimensions + 1 bias row
source_with_bias = np.column_stack(
    [source_embeddings, np.ones(source_embeddings.shape[0])]
)
target_embeddings = source_with_bias @ true_matrix

# Learn the FEATMAP transformation from the paired embeddings.
transformation_matrix = train_FEATMAP(source_embeddings, target_embeddings)

# Apply it to new embeddings.
new_embeddings = rng.normal(size=(10, 8))
transformed_embeddings = apply_FEATMAP(new_embeddings, transformation_matrix)

print("Learned matrix shape:", tuple(transformation_matrix.shape))
print("Input shape:", new_embeddings.shape)
print("Output shape:", transformed_embeddings.shape)
