import numpy as np
import torch


from core.utils import cross_product, get_discrete_fvals, get_indices_from_ref_array
from core.uncertainty import compute_unc_objective


def compute_simple_regret(
    obj_func, decision_points, context_points, ref_dist, chosen_X, config
):
    # Compute for all points
    joint_points = cross_product(decision_points, context_points)
    discrete_fvals = get_discrete_fvals(
        fvals=obj_func(joint_points),
        decision_points=decision_points,
        context_points=context_points,
    )
    all_unc_obj_vals = compute_unc_objective(
        discrete_fvals=discrete_fvals,
        ref_dist=ref_dist,
        distance_name=config.distance_name,
        alpha=config.alpha,
        eps_1=config.eps_1,
        eps_2=config.eps_2,
        h=0.00001,  # use extra accurate one here
    )
    chosen_idxs = get_indices_from_ref_array(input=chosen_X, ref=decision_points)
    chosen_vals = all_unc_obj_vals[chosen_idxs]
    max_val = np.max(all_unc_obj_vals)
    simple_regret = (
        (max_val - torch.cummax(torch.tensor(chosen_vals), dim=0)[0])
        .cpu()
        .detach()
        .numpy()
    )

    return simple_regret
