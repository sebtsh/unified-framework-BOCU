import numpy as np

from core.uncertainty import compute_unc_objective
from core.utils import cross_product, get_discrete_fvals


def acquire(gp, decision_points, context_points, ref_dist, config):
    acquisition = config.acquisition

    if acquisition == "ts":
        best_idx = thompson_sampling(
            gp=gp,
            decision_points=decision_points,
            context_points=context_points,
            ref_dist=ref_dist,
            config=config,
        )
    else:
        raise NotImplementedError

    return decision_points[best_idx][None, :]


def thompson_sampling(gp, decision_points, context_points, ref_dist, config):
    gp.eval()
    joint_points = cross_product(decision_points, context_points)
    pred = gp(joint_points)
    fvals = pred.sample()  # (len(points), )
    discrete_fvals = get_discrete_fvals(
        fvals=fvals, decision_points=decision_points, context_points=context_points
    )

    unc_obj_vals = compute_unc_objective(
        discrete_fvals=discrete_fvals,
        ref_dist=ref_dist,
        distance_name=config.distance_name,
        alpha=config.alpha,
        eps_1=config.eps_1,
        eps_2=config.eps_2,
        h=config.finite_diff_h,
    )

    best_idx = np.argmax(unc_obj_vals)

    return best_idx
