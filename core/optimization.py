import torch

from core.acquisition import acquire
from core.model import ExactGPModel
from core.utils import log


def bo_loop(
    train_X,
    train_y,
    decision_points,
    context_points,
    kernel,
    likelihood,
    noisy_obj_func,
    config,
):
    for t in range(config.T):
        log(f"Iteration {t}")
        gp = ExactGPModel(
            train_x=train_X,
            train_y=torch.squeeze(train_y),
            kernel=kernel,
            likelihood=likelihood,
        )

        x_t = acquire(gp, decision_points, context_points, config)  # (1, d)
        y_t = noisy_obj_func(x_t)
        train_X = torch.cat([train_X, x_t], dim=0)
        train_y = torch.cat([train_y, y_t], dim=0)

    return train_X, train_y
