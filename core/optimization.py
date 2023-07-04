import torch
from tqdm import trange

from core.acquisition import acquire
from core.model import ExactGPModel
from core.utils import log


def bo_loop(
    train_Z,
    train_y,
    decision_points,
    context_points,
    kernel,
    likelihood,
    noisy_obj_func,
    ref_dist,
    true_dist,
    config,
):
    chosen_X = []
    for t in trange(config.T):
        # log(f"Iteration {t}")
        gp = ExactGPModel(
            train_x=train_Z,
            train_y=torch.squeeze(train_y),
            kernel=kernel,
            likelihood=likelihood,
        )

        x_t = acquire(
            gp=gp,
            decision_points=decision_points,
            context_points=context_points,
            ref_dist=ref_dist,
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
