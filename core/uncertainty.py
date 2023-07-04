import cvxpy as cp
import numpy as np
import torch


def compute_unc_objective(
    discrete_fvals, ref_dist, distance_name, alpha, eps_1, eps_2, h
):
    """
    Computes g(x) = alpha * v_x(eps_1) + \delta_x(eps_1) * eps_2, where v_x is the distributionally robust value
    and \delta_x is the right derivative of v_x.
    :param discrete_fvals: Array of shape (|D|, |C|) where D is the decision variable set and C is the context variable
    set.
    :param ref_dist: Probability distribution represented as an array of shape (|C|).
    :param distance_name: Distribution distance. String.
    :param alpha: float.
    :param eps_1: float.
    :param eps_2: float.
    :param h: float. Finite difference amount.
    :return:
    """
    assert alpha > 0 or eps_2 > 0
    v_x = compute_dr_values(
        discrete_fvals=discrete_fvals,
        ref_dist=ref_dist,
        distance_name=distance_name,
        eps=eps_1,
    )

    if eps_2 > 0:
        v_x_plus_h = compute_dr_values(
            discrete_fvals=discrete_fvals,
            ref_dist=ref_dist,
            distance_name=distance_name,
            eps=eps_1 + h,
        )

        delta_x = (v_x_plus_h - v_x) / h
    else:
        delta_x = 0

    return alpha * v_x + delta_x * eps_2


def compute_dr_values(discrete_fvals, ref_dist, distance_name, eps):
    """
    Computes distributionally robust values.
    :param discrete_fvals: Array of shape (|D|, |C|) where D is the decision variable set and C is the context variable
    set.
    :param ref_dist: Probability distribution represented as an array of shape (|C|).
    :param distance_name: Distribution distance. String.
    :param eps: Margin, float.
    :return: Array of shape (|D|, ).
    """
    cvx_prob = create_cvx_prob(p=ref_dist, distance_name=distance_name, eps=eps)
    dr_vals = []
    for i in range(len(discrete_fvals)):
        dr_val, _ = cvx_prob(discrete_fvals[i].cpu().detach().numpy())
        dr_vals.append(dr_val)

    return np.array(dr_vals)


def create_cvx_prob(p, distance_name, eps):
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

    if distance_name == "tv":
        constraints = [cp.sum(q) == 1.0, q >= 0.0, cp.norm(p - q, 1) <= eps]
    else:
        raise NotImplementedError

    prob = cp.Problem(objective, constraints)

    def wrapper(f):
        g.value = f
        try:
            value = prob.solve(warm_start=True)
        except:
            print("Default solver failed, trying SCS")
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


def tv(p, q):
    """
    Calculates the total variation distance between 2 discrete distributions.
    :param p: array of shape (|C|, )
    :param q: array of shape (|C|, )
    :return: float
    """
    return torch.linalg.norm(p - q, ord=1)
