import pickle
import torch

from config import get_config, set_dir_attributes
from core.metrics import compute_regret
from core.objectives import get_objective
from core.optimization import bo_loop
from core.uncertainty import create_cvx_prob, get_discrete_uniform_dist
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
    kernel = create_kernel(
        dims=config.decision_dims + config.context_dims,
        kernel_name=config.kernel,
        config=config,
    )
    likelihood = create_likelihood(config)
    obj_func, noisy_obj_func = get_objective(
        kernel=kernel, bounds=joint_bounds, config=config
    )

    # Get reference and true distribution
    ref_dist = get_discrete_uniform_dist(context_points=context_points)
    # TODO: change true_dist, maybe make sure is within margin
    true_dist = get_discrete_uniform_dist(context_points=context_points)

    # Create cvxpy problems. WARNING: currently assumes reference distribution and margin is the same for all
    # iterations. If not true, new cvxpy problems must be created at every iteration
    if config.distance_name == "mmd":
        mmd_kernel = create_kernel(
            dims=config.context_dims, kernel_name=config.kernel, config=config
        )
    else:
        mmd_kernel = None
    cvx_prob = create_cvx_prob(
        p=ref_dist.cpu().detach().numpy(),
        distance_name=config.distance_name,
        eps=config.eps_1,
        context_points=context_points,
        mmd_kernel=mmd_kernel,
        jitter=config.jitter,
    )
    cvx_prob_plus_h = create_cvx_prob(
        p=ref_dist.cpu().detach().numpy(),
        distance_name=config.distance_name,
        eps=config.eps_1 + config.finite_diff_h,
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
        true_dist=true_dist,
        cvx_prob=cvx_prob,
        cvx_prob_plus_h=cvx_prob_plus_h,
        config=config,
    )

    # Calculate regret
    simple_regret, cumu_regret = compute_regret(
        obj_func=obj_func,
        decision_points=decision_points,
        context_points=context_points,
        cvx_prob=cvx_prob,
        cvx_prob_plus_h=cvx_prob_plus_h,
        chosen_X=chosen_X,
        config=config,
    )

    # Save results
    pickle.dump(
        (simple_regret, cumu_regret),
        open(config.pickles_save_dir + config.filename + ".p", "wb"),
    )

    log("Run complete")


if __name__ == "__main__":
    run_exp(get_config(add_compulsory_args=True))
