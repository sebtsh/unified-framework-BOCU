import numpy as np
import torch


from core.utils import cross_product, get_discrete_fvals, get_indices_from_ref_array
from core.uncertainty import compute_unc_objective


def compute_regret(
    obj_func,
    decision_points,
    context_points,
    cvx_prob,
    cvx_prob_plus_h,
    chosen_X,
    config,
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
        cvx_prob=cvx_prob,
        cvx_prob_plus_h=cvx_prob_plus_h,
        alpha=config.alpha,
        eps_2=config.eps_2,
        h=config.finite_diff_h,
    )
    chosen_idxs = get_indices_from_ref_array(input=chosen_X, ref=decision_points)
    chosen_vals = all_unc_obj_vals[chosen_idxs]
    max_val = np.max(all_unc_obj_vals)
    simple_regret = np.squeeze(
        (max_val - torch.cummax(torch.tensor(chosen_vals), dim=0)[0])
        .cpu()
        .detach()
        .numpy()
    )

    cumu_regret = np.squeeze(np.cumsum(max_val - chosen_vals))

    return simple_regret, cumu_regret
