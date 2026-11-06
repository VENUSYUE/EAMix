"""EAMix: Coupled Dual-Layer Edge Random Walk for Attributed Edge Clustering."""

from .clustering import InsufficientRepresentation, correct_assignments, spectral_coordinates
from .metrics import evaluate
from .model import EAMix, EAMixConfig
from .relations import (TopologyOperator, attribute_affinity, attribute_transition,
                        endpoint_descriptors, row_normalize, shared_signal)
from .walk import CoupledWalk, fuse, terminal_responses

__all__ = ["EAMix", "EAMixConfig", "InsufficientRepresentation", "CoupledWalk", "TopologyOperator",
           "attribute_affinity", "attribute_transition", "correct_assignments", "endpoint_descriptors",
           "evaluate", "fuse", "row_normalize", "shared_signal", "spectral_coordinates",
           "terminal_responses"]
