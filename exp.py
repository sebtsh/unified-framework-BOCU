import numpy as np
import torch

from config import get_config, set_dir_attributes
from core.objectives import get_objective
from core.optimization import bo_loop
from core.utils import (
    construct_bounds,
    construct_grid,
    create_kernel,
    create_likelihood,
    cross_product,
    log,
)


def run_exp(config):
    log(f"======== NEW RUN ========")
    config = set_dir_attributes(config)
    for arg in vars(config):
        print(f"{arg}: {getattr(config, arg)}")
    torch.manual_seed(config.seed)
    torch.set_default_dtype(torch.float64)

    # Construct spaces
    decision_bounds = construct_bounds(lower=0.0, upper=1.0, d=config.decision_dims)
    context_bounds = construct_bounds(lower=0.0, upper=1.0, d=config.context_dims)
    joint_bounds = torch.cat([decision_bounds, context_bounds], dim=-1)
    decision_points = construct_grid(
        bounds=decision_bounds, density_per_dim=config.decision_density_per_dim
    )
    context_points = construct_grid(
        bounds=context_bounds, density_per_dim=config.context_density_per_dim
    )
    joint_points = cross_product(decision_points, context_points)

    # Get objective function
    kernel = create_kernel(config)
    likelihood = create_likelihood(config)
    obj_func, noisy_obj_func = get_objective(
        kernel=kernel, bounds=joint_bounds, config=config
    )

    # Get initial observations
    init_X = joint_points[torch.randperm(len(joint_points))[: config.num_init_points]]
    init_y = noisy_obj_func(init_X)

    # Main BO loop
    final_X, final_y = bo_loop(
        train_X=init_X,
        train_y=init_y,
        decision_points=decision_points,
        context_points=context_points,
        kernel=kernel,
        likelihood=likelihood,
        noisy_obj_func=noisy_obj_func,
        config=config,
    )

    # Calculate regret TODO: change to stochastic version
    chosen_X = final_X[config.num_init_points :]
    chosen_vals = obj_func(chosen_X)
    max_val = torch.max(obj_func(joint_points))
    simple_regret = (
        (max_val - torch.cummax(chosen_vals, dim=0)[0]).cpu().detach().numpy()
    )
    print(np.squeeze(simple_regret))


if __name__ == "__main__":
    run_exp(get_config(add_compulsory_args=True))
