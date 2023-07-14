import pickle
import torch

from core.utils import uniform_samples
from data.plant.plant_funcs import create_leaf_max_area_func


def get_objective(kernel, bounds, config):
    task = config.task

    if task == "gp":
        obj_func = sample_gp_prior(
            kernel=kernel,
            bounds=bounds,
            num_points=config.gp_sample_num_points,
            jitter=config.jitter,
        )
    elif task == "plant":
        bounds = torch.tensor(
            [[0, 7.7], [0, 3.5], [0, 10.4], [8.9, 11.3], [2.5, 6.5]], dtype=torch.double
        ).T
        leafarea_meanvar_func = create_leaf_max_area_func(standardize=True)
        obj_func = lambda x: torch.tensor(leafarea_meanvar_func(x.numpy())[0])
        obj_func = input_transform_wrapper(obj_func=obj_func, bounds=bounds)
    elif task == "infection":
        X, y = pickle.load(open("data/infection/infection_X_y.p", "rb"))
        X_torch = torch.tensor(X)
        y_torch = torch.tensor(y)
        obj_func = gp_mean_from_samples(
            kernel=kernel, X=X_torch, f=y_torch, jitter=config.jitter
        )
    else:
        raise NotImplementedError

    noisy_obj_func = noisy_wrapper(obj_func=obj_func, noise_std=config.noise_std)

    return obj_func, noisy_obj_func


def sample_gp_prior(kernel, bounds, num_points, jitter):
    """
    Sample a random function from a GP prior with mean 0 and covariance specified by a kernel.
    :param kernel: a GPyTorch kernel.
    :param bounds: array of shape (2, num_dims).
    :param num_points: int.
    :param jitter: float.
    :return: Callable that takes in an array of shape (n, N) and returns an array of shape (n, 1).
    """
    points = uniform_samples(bounds=bounds, n_samples=num_points)
    cov = kernel(points).evaluate() + jitter * torch.eye(num_points)
    f_vals = torch.distributions.MultivariateNormal(
        torch.zeros(num_points, dtype=torch.double), cov
    ).sample()[:, None]

    return gp_mean_from_samples(kernel=kernel, X=points, f=f_vals, jitter=jitter)


def gp_mean_from_samples(kernel, X, f, jitter):
    cov = kernel(X).evaluate() + jitter * torch.eye(len(X))
    L = torch.linalg.cholesky(cov)
    L_bs_f = torch.linalg.solve_triangular(L, f, upper=False)
    LT_bs_L_bs_f = torch.linalg.solve_triangular(L.T, L_bs_f, upper=True)

    return lambda x: kernel(x, X).evaluate() @ LT_bs_L_bs_f


def noisy_wrapper(obj_func, noise_std):
    """
    Wrapper around an existing objective function. Turns a noiseless objective function into a noisy one.
    :param obj_func: Callable that takes in an array of shape (..., d) and returns an array of shape (..., 1).
    :param noise_std: float.
    :return: Callable that takes in an array of shape (..., d) and returns an array of shape (..., 1).
    """
    return lambda x: obj_func(x) + noise_std * torch.randn(size=x.shape[:-1] + (1,))


def input_transform_wrapper(obj_func, bounds):
    """
    Wrapper around an existing objective function. Changes the bounds of the objective function to be the d-dim
    unit hypercube [0, 1]^d.
    :param obj_func: Callable that takes in an array of shape (..., d) and returns an array of shape (..., 1).
    :param bounds: array of shape (2, d).
    :return: Callable that takes in an array of shape (..., d) and returns an array of shape (..., 1).
    """
    return lambda x: obj_func(input_transform(x, bounds))


def input_transform(x, bounds):
    return x * (bounds[1] - bounds[0]) + bounds[0]
