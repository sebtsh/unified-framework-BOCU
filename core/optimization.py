import torch

from core.acquisition import acquire
from core.utils import log


def bo_loop(
    train_Z,
    train_y,
    decision_points,
    context_points,
    kernel,
    likelihood,
    noisy_obj_func,
    true_dist,
    cvx_prob,
    cvx_prob_plus_h,
    config,
):
    chosen_X = []
    for t in range(config.T):
        log(f"Iteration {t}")

        x_t = acquire(
            train_X=train_Z,
            train_y=train_y,
            likelihood=likelihood,
            kernel=kernel,
            decision_points=decision_points,
            context_points=context_points,
            cvx_prob=cvx_prob,
            cvx_prob_plus_h=cvx_prob_plus_h,
            config=config,
        )  # (1, d)

        c_t_idx = torch.multinomial(input=true_dist, num_samples=1)[0]
        c_t = context_points[c_t_idx][None, :]
        z_t = torch.cat([x_t, c_t], dim=-1)
        y_t = noisy_obj_func(z_t)

        chosen_X.append(x_t)
        train_Z = torch.cat([train_Z, z_t], dim=0)
        train_y = torch.cat([train_y, y_t], dim=0)

    chosen_X = torch.cat(chosen_X, dim=0)

    return chosen_X, train_Z, train_y
