import numpy as np
import pickle
import torch

from config import (
    get_config,
    set_dir_attributes,
    set_task_attributes,
    set_unc_attributes,
)
from core.metrics import compute_regret
from core.objectives import get_objective
from core.optimization import bo_loop
from core.uncertainty import (
    compute_distance,
    create_cvx_prob,
    get_discrete_normal_dist,
    get_discrete_uniform_dist,
)
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
    config = set_task_attributes(config)
    config = set_unc_attributes(config)

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
    kernel = create_kernel(
        dims=config.decision_dims + config.context_dims,
        kernel_name=config.kernel,
        config=config,
    )
    likelihood = create_likelihood(config)
    obj_func, noisy_obj_func = get_objective(
        kernel=kernel, bounds=joint_bounds, config=config
    )

    # Get reference and true distribution, and margin eps
    ref_mean = config.ref_mean * np.ones(config.context_dims)
    ref_cov = config.ref_var * np.eye(config.context_dims)
    ref_dist = get_discrete_normal_dist(
        context_points=context_points, mean=ref_mean, cov=ref_cov
    )
    true_dist = get_discrete_uniform_dist(context_points=context_points)
    if config.distance_name == "mmd":
        mmd_kernel = create_kernel(
            dims=config.context_dims, kernel_name=config.kernel, config=config
        )
        M = mmd_kernel(context_points)
    else:
        mmd_kernel = None
        M = None
    if config.unc_obj == "wcs":
        eps = 0.0
    else:
        eps = compute_distance(
            p=ref_dist, q=true_dist, M=M, distance_name=config.distance_name
        )

    # Create cvxpy problems. WARNING: currently assumes reference distribution and margin is the same for all
    # iterations. If not true, new cvxpy problems must be created at every iteration
    cvx_prob = create_cvx_prob(
        p=ref_dist.cpu().detach().numpy(),
        distance_name=config.distance_name,
        eps=eps,
        context_points=context_points,
        mmd_kernel=mmd_kernel,
        jitter=config.jitter,
    )
    cvx_prob_plus_h = create_cvx_prob(
        p=ref_dist.cpu().detach().numpy(),
        distance_name=config.distance_name,
        eps=eps + config.finite_diff_h,
        context_points=context_points,
        mmd_kernel=mmd_kernel,
        jitter=config.jitter,
    )

    # Get initial observations
    init_Z = joint_points[torch.randperm(len(joint_points))[: config.num_init_points]]
    init_y = noisy_obj_func(init_Z)

    # Main BO loop
    chosen_X, _, _ = bo_loop(
        train_Z=init_Z,
        train_y=init_y,
        decision_points=decision_points,
        context_points=context_points,
        kernel=kernel,
        likelihood=likelihood,
        noisy_obj_func=noisy_obj_func,
        ref_dist=ref_dist,
        true_dist=true_dist,
        cvx_prob=cvx_prob,
        cvx_prob_plus_h=cvx_prob_plus_h,
        config=config,
    )

    # Calculate regret wrt approximate objective
    simple_regret_approx, cumu_regret_approx = compute_regret(
        obj_func=obj_func,
        decision_points=decision_points,
        context_points=context_points,
        cvx_prob=cvx_prob,
        cvx_prob_plus_h=cvx_prob_plus_h,
        h=config.finite_diff_h,
        chosen_X=chosen_X,
        config=config,
    )

    # Calculate regret wrt to "true" objective (more accurate via smaller h)
    cvx_prob_plus_h_reduced = create_cvx_prob(
        p=ref_dist.cpu().detach().numpy(),
        distance_name=config.distance_name,
        eps=eps + config.finite_diff_h * 1e-02,
        context_points=context_points,
        mmd_kernel=mmd_kernel,
        jitter=config.jitter,
    )
    simple_regret_true, cumu_regret_true = compute_regret(
        obj_func=obj_func,
        decision_points=decision_points,
        context_points=context_points,
        cvx_prob=cvx_prob,
        cvx_prob_plus_h=cvx_prob_plus_h_reduced,
        h=config.finite_diff_h * 1e-02,
        chosen_X=chosen_X,
        config=config,
    )

    # Save results
    pickle.dump(
        (
            simple_regret_approx,
            cumu_regret_approx,
            simple_regret_true,
            cumu_regret_true,
        ),
        open(config.pickles_save_dir + config.filename + ".p", "wb"),
    )

    log("Run complete")


if __name__ == "__main__":
    run_exp(get_config(add_compulsory_args=True))
