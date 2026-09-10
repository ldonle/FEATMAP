"""Examples using the provided FEATMAP transformation matrices.

The examples use random vectors only to demonstrate the API. They are not
meaningful medical embeddings.
"""

import sys
from pathlib import Path

import torch

# Allow this file to be run directly from the repository without installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from featmap import (
    harmonize_field_strength,
    harmonize_foundation_model,
    harmonize_scanner,
)


# Scanner harmonization: UNI embeddings are 1024-dimensional.
uni_embeddings = torch.randn(4, 1024)
scanner_harmonized = harmonize_scanner(
    uni_embeddings,
    source_scanner="at2",
    target_scanner="gt450",
    foundation_model="uni",
)
print("Scanner harmonization:", tuple(scanner_harmonized.shape))

# Foundation-model harmonization: UNI (1024) -> CONCH (512).
model_harmonized = harmonize_foundation_model(
    uni_embeddings,
    source_model="uni",
    target_model="conch",
    scanner="at2",
)
print("Foundation-model harmonization:", tuple(model_harmonized.shape))

# MRI field-strength harmonization: BrainIAC embeddings are 768-dimensional.
brainiac_embeddings = torch.randn(4, 768)
field_strength_harmonized = harmonize_field_strength(
    brainiac_embeddings,
    source_field_strength="1p5t",
    target_field_strength="3t",
)
print("Field-strength harmonization:", tuple(field_strength_harmonized.shape))
