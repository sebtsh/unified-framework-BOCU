import cvxpy as cp
import numpy as np
from scipy.stats import multivariate_normal
import torch

from core.utils import log


def compute_unc_objective(discrete_fvals, cvx_prob, cvx_prob_plus_h, alpha, beta, h):
    """
    Computes g(x) = alpha * v_x(eps) + beta * delta_x(eps), where v_x is the distributionally robust value
    and delta_x is the right derivative of v_x.
    :param discrete_fvals: Array of shape (|D|, |C|) where D is the decision variable set and C is the context variable
    set.
    :param cvx_prob:
    :param cvx_prob_plus_h:
    :param alpha: float.
    :param beta: float.
    :param h: float. Finite difference amount.
    :return:
    """
    assert alpha > 0 or beta > 0
    v_x = compute_dr_values(
        discrete_fvals=discrete_fvals,
        cvx_prob=cvx_prob,
    )

    if beta > 0:
        v_x_plus_h = compute_dr_values(
            discrete_fvals=discrete_fvals, cvx_prob=cvx_prob_plus_h
        )

        delta_x = (v_x_plus_h - v_x) / h
    else:
        delta_x = 0

    return alpha * v_x + beta * delta_x


def compute_unc_objective_ucb(
    discrete_ucb_vals, discrete_lcb_vals, cvx_prob, cvx_prob_plus_h, alpha, beta, h
):
    """
    Same as compute_unc_objective except using upper and lower confidence bounds as functions.
    :param discrete_ucb_vals: Array of shape (|D|, |C|) where D is the decision variable set and C is the context
    variable set.
    :param discrete_lcb_vals: Array of shape (|D|, |C|) where D is the decision variable set and C is the context
    variable set.
    :param cvx_prob:
    :param cvx_prob_plus_h:
    :param alpha: float.
    :param beta: float.
    :param h: float. Finite difference amount.
    :return:
    """
    assert alpha > 0 or beta > 0
    v_x_ucb = compute_dr_values(
        discrete_fvals=discrete_ucb_vals,
        cvx_prob=cvx_prob,
    )

    if beta > 0:
        v_x_plus_h_ucb = compute_dr_values(
            discrete_fvals=discrete_ucb_vals, cvx_prob=cvx_prob_plus_h
        )
        v_x_lcb = compute_dr_values(discrete_fvals=discrete_lcb_vals, cvx_prob=cvx_prob)

        delta_x = (v_x_plus_h_ucb - v_x_lcb) / h
    else:
        delta_x = 0

    return alpha * v_x_ucb + beta * delta_x


def compute_dr_values(discrete_fvals, cvx_prob):
    """
    Computes distributionally robust values.
    :param discrete_fvals: Array of shape (|D|, |C|) where D is the decision variable set and C is the context variable
    set.
    :param cvx_prob:
    :return: Array of shape (|D|, ).
    """
    dr_vals = []
    for i in range(len(discrete_fvals)):
        dr_val, _ = cvx_prob(discrete_fvals[i].cpu().detach().numpy())
        dr_vals.append(dr_val)

    return np.array(dr_vals)


def create_cvx_prob(p, distance_name, eps, context_points, mmd_kernel, jitter):
    """

    :param p: reference distribution.
    :param distance:
    :param eps:
    :return:
    """
    num_context = len(p)

    q = cp.Variable(num_context)
    g = cp.Parameter(num_context)

    objective = cp.Minimize(q @ g)

    constraints = [cp.sum(q) == 1.0, q >= 0.0]
    if distance_name == "tv":
        constraints.append(cp.norm(p - q, 1) <= eps)
    elif distance_name == "mmd":
        M = (
            mmd_kernel(context_points).evaluate().cpu().detach().numpy()
            + np.eye(len(context_points)) * jitter
        )
        L = np.linalg.cholesky(M)
        assert np.allclose(L @ L.T, M)
        constraints.append(cp.norm(L.T @ (p - q), 2) <= eps)
    else:
        raise NotImplementedError

    prob = cp.Problem(objective, constraints)

    def wrapper(f):
        g.value = f
        try:
            value = prob.solve(warm_start=True)
        except:
            log("Default solver failed, trying SCS")
            value = prob.solve(solver="SCS", warm_start=True)
        sol = q.value

        return value, sol

    return wrapper


def get_discrete_uniform_dist(context_points):
    """
    Returns an array of shape |C| that is a uniform probability distribution over the context set.
    :param context_points: Array of shape (|C|, 1)
    :return: array of shape (|C|, )
    """
    return torch.ones(len(context_points)) * (1 / len(context_points))


def get_discrete_normal_dist(context_points, mean, cov):
    """
    Returns an array of shape |C| that is a probability distribution over the context set. Uses the normal distribution
    with the specified mean and variance.
    :param context_points: Array of shape (|C|, d)
    :param mean: array of shape (d, )
    :param cov: array of shape (d, d)
    :return: array of shape (|C|, )
    """
    rv = multivariate_normal(mean=mean, cov=cov, allow_singular=False)
    pdfs = rv.pdf(context_points)
    return torch.tensor(pdfs / np.sum(pdfs))


def tv(p, q):
    """
    Calculates the total variation distance between 2 discrete distributions.
    :param p: array of shape (|C|, )
    :param q: array of shape (|C|, )
    :return: float
    """
    return torch.linalg.norm(p - q, ord=1).item()


def mmd(p, q, M):
    v = p - q
    return torch.sqrt(v @ M @ v).item()


def check_psd(A):
    # check symmetric
    assert np.allclose(A, A.T)
    # check eigenvalues
    eigvals = np.linalg.eigvalsh(A)
    assert np.min(eigvals) > 0
    # probably not necessary but
    sign, logdet = np.linalg.slogdet(A)
    assert np.allclose(sign, 1.0)


def compute_distance(p, q, M, distance_name):
    if distance_name == "tv":
        return tv(p, q)
    elif distance_name == "mmd":
        return mmd(p, q, M)
    else:
        raise NotImplementedError
