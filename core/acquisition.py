from botorch.models import SingleTaskGP
from contextlib import ExitStack
import copy
import gpytorch.settings as gpts
import numpy as np
import torch

from config import set_unc_attributes
from core.uncertainty import compute_unc_objective, compute_unc_objective_ucb_naive
from core.utils import create_kernel, cross_product, get_discrete_fvals


def acquire(
    train_X,
    train_y,
    likelihood,
    kernel,
    decision_points,
    context_points,
    ref_dist,
    cvx_prob,
    cvx_prob_plus_h,
    config,
):
    acquisition = config.acquisition

    if acquisition == "random":
        best_idx = random(decision_points=decision_points)
    elif acquisition == "ts":
        assert (
            config.kernel == "se"
        )  # otherwise this RFF kernel is using the wrong features
        rff_kernel = create_kernel(
            dims=config.decision_dims + config.context_dims,
            kernel_name="rff",
            config=config,
        )
        gp = SingleTaskGP(
            train_X=train_X,
            train_Y=train_y,
            likelihood=likelihood,
            covar_module=rff_kernel,
        )

        best_idx = thompson_sampling(
            gp=gp,
            decision_points=decision_points,
            context_points=context_points,
            cvx_prob=cvx_prob,
            cvx_prob_plus_h=cvx_prob_plus_h,
            config=config,
        )
    elif acquisition == "ucb":
        gp = SingleTaskGP(
            train_X=train_X, train_Y=train_y, likelihood=likelihood, covar_module=kernel
        )

        best_idx = ucb_naive(
            gp=gp,
            decision_points=decision_points,
            context_points=context_points,
            cvx_prob=cvx_prob,
            cvx_prob_plus_h=cvx_prob_plus_h,
            config=config,
        )
    elif acquisition == "ucbu":
        gp = SingleTaskGP(
            train_X=train_X, train_Y=train_y, likelihood=likelihood, covar_module=kernel
        )

        best_idx = ucb_unjust(
            gp=gp,
            decision_points=decision_points,
            context_points=context_points,
            cvx_prob=cvx_prob,
            cvx_prob_plus_h=cvx_prob_plus_h,
            config=config,
        )
    elif acquisition == "so":
        gp = SingleTaskGP(
            train_X=train_X, train_Y=train_y, likelihood=likelihood, covar_module=kernel
        )

        best_idx = so(
            gp=gp,
            decision_points=decision_points,
            context_points=context_points,
            ref_dist=ref_dist,
            config=config,
        )
    elif acquisition == "ro":
        gp = SingleTaskGP(
            train_X=train_X, train_Y=train_y, likelihood=likelihood, covar_module=kernel
        )

        best_idx = ro(
            gp=gp,
            decision_points=decision_points,
            context_points=context_points,
            config=config,
        )
    elif acquisition in ["tsdro", "tswcs", "tsgen"]:
        # hack to force using TS-BOCU with the wrong hyperparameters of alpha, beta, and eps.
        # Only for results in Appendix.
        assert (
            config.kernel == "se"
        )  # otherwise this RFF kernel is using the wrong features
        rff_kernel = create_kernel(
            dims=config.decision_dims + config.context_dims,
            kernel_name="rff",
            config=config,
        )
        gp = SingleTaskGP(
            train_X=train_X,
            train_Y=train_y,
            likelihood=likelihood,
            covar_module=rff_kernel,
        )

        wrong_config = copy.deepcopy(config)
        acq_unc_obj = acquisition[-3:]
        wrong_config.unc_obj = acq_unc_obj
        wrong_config = set_unc_attributes(wrong_config)

        best_idx = thompson_sampling(
            gp=gp,
            decision_points=decision_points,
            context_points=context_points,
            cvx_prob=cvx_prob,
            cvx_prob_plus_h=cvx_prob_plus_h,
            config=wrong_config,
        )
    else:
        raise NotImplementedError

    return decision_points[best_idx][None, :]


def thompson_sampling(
    gp, decision_points, context_points, cvx_prob, cvx_prob_plus_h, config
):
    gp.eval()
    joint_points = cross_product(decision_points, context_points)
    # pred = gp(joint_points)
    # fvals = pred.sample()  # (len(points), )

    with ExitStack() as es:
        # RFF settings
        es.enter_context(gpts.fast_computations(covar_root_decomposition=True))

    with torch.no_grad():
        posterior = gp.posterior(joint_points)
        fvals = posterior.rsample().squeeze([0, -1])

    discrete_fvals = get_discrete_fvals(
        fvals=fvals, decision_points=decision_points, context_points=context_points
    )

    unc_obj_vals = compute_unc_objective(
        discrete_fvals=discrete_fvals,
        cvx_prob=cvx_prob,
        cvx_prob_plus_h=cvx_prob_plus_h,
        alpha=config.alpha,
        beta=config.beta,
        h=config.finite_diff_h,
    )

    best_idx = np.argmax(unc_obj_vals)

    return best_idx


def so(gp, decision_points, context_points, ref_dist, config):
    gp.eval()
    joint_points = cross_product(decision_points, context_points)

    pred = gp(joint_points)
    mean = pred.mean
    variance = pred.variance
    ucb_vals = mean + config.beta * torch.sqrt(variance)
    discrete_ucb_vals = get_discrete_fvals(
        fvals=ucb_vals, decision_points=decision_points, context_points=context_points
    )
    obj_vals = (discrete_ucb_vals @ ref_dist).cpu().detach().numpy()
    best_idx = np.argmax(obj_vals)

    return best_idx


def ro(gp, decision_points, context_points, config):
    gp.eval()
    joint_points = cross_product(decision_points, context_points)

    pred = gp(joint_points)
    mean = pred.mean
    variance = pred.variance
    ucb_vals = mean + config.beta * torch.sqrt(variance)
    discrete_ucb_vals = (
        get_discrete_fvals(
            fvals=ucb_vals,
            decision_points=decision_points,
            context_points=context_points,
        )
        .cpu()
        .detach()
        .numpy()
    )
    obj_vals = np.min(discrete_ucb_vals, axis=-1)
    best_idx = np.argmax(obj_vals)

    return best_idx


def ucb_naive(gp, decision_points, context_points, cvx_prob, cvx_prob_plus_h, config):
    gp.eval()
    joint_points = cross_product(decision_points, context_points)

    pred = gp(joint_points)
    mean = pred.mean
    variance = pred.variance
    ucb_vals = mean + config.beta * torch.sqrt(variance)
    lcb_vals = mean - config.beta * torch.sqrt(variance)
    discrete_ucb_vals = get_discrete_fvals(
        fvals=ucb_vals, decision_points=decision_points, context_points=context_points
    )
    discrete_lcb_vals = get_discrete_fvals(
        fvals=lcb_vals, decision_points=decision_points, context_points=context_points
    )

    unc_obj_vals = compute_unc_objective_ucb_naive(
        discrete_ucb_vals=discrete_ucb_vals,
        discrete_lcb_vals=discrete_lcb_vals,
        cvx_prob=cvx_prob,
        cvx_prob_plus_h=cvx_prob_plus_h,
        alpha=config.alpha,
        beta=config.beta,
        h=config.finite_diff_h,
    )

    best_idx = np.argmax(unc_obj_vals)

    return best_idx


def ucb_unjust(gp, decision_points, context_points, cvx_prob, cvx_prob_plus_h, config):
    gp.eval()
    joint_points = cross_product(decision_points, context_points)

    pred = gp(joint_points)
    mean = pred.mean
    variance = pred.variance
    ucb_vals = mean + config.beta * torch.sqrt(variance)
    discrete_ucb_vals = get_discrete_fvals(
        fvals=ucb_vals, decision_points=decision_points, context_points=context_points
    )

    unc_obj_vals = compute_unc_objective(
        discrete_fvals=discrete_ucb_vals,
        cvx_prob=cvx_prob,
        cvx_prob_plus_h=cvx_prob_plus_h,
        alpha=config.alpha,
        beta=config.beta,
        h=config.finite_diff_h,
    )

    best_idx = np.argmax(unc_obj_vals)

    return best_idx


def random(decision_points):
    return torch.randint(high=len(decision_points), size=(1,))[0].item()
