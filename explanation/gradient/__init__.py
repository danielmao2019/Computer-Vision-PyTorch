"""
GRADIENT APIs.
"""
from explanation.gradient import gradients
from explanation.gradient import hooks
from explanation.gradient.gradient_wrt_inputs import get_backward_model
from explanation.gradient.gradient_wrt_inputs import compute_gradients


__all__ = (
    "hooks",
    "gradients",
    "get_backward_model",
    "compute_gradients",
)
