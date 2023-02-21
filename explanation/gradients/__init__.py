"""
GRADIENT APIs.
"""
from explanation.gradients.gradients import tanh_gradient, CE_gradient, MSE_gradient
from explanation.gradients import hooks
from explanation.gradients.gradient_wrt_inputs import get_backward_model


__all__ = (
    "hooks",
    "tanh_gradient", "CE_gradient", "MSE_gradient",
    "get_backward_model",
)
