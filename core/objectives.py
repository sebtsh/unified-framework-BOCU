import torch

from core.utils import uniform_samples


def get_objective(kernel, bounds, config):
    task = config.task

    if task == "gp":
        obj_func = sample_gp_prior(
            kernel=kernel, bounds=bounds, num_points=config.gp_sample_num_points
        )
    else:
        raise NotImplementedError

    noisy_obj_func = noisy_wrapper(obj_func=obj_func, noise_std=config.noise_std)

    return obj_func, noisy_obj_func


def sample_gp_prior(kernel, bounds, num_points, jitter=1e-06):
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

    L = torch.linalg.cholesky(cov)
    L_bs_f = torch.linalg.solve_triangular(L, f_vals, upper=False)
    LT_bs_L_bs_f = torch.linalg.solve_triangular(L.T, L_bs_f, upper=True)
    return lambda x: kernel(x, points).evaluate() @ LT_bs_L_bs_f


def noisy_wrapper(obj_func, noise_std):
    """
    Wrapper around an existing objective function. Turns a noiseless objective function into a noisy one.
    :param obj_func: Callable that takes in an array of shape (..., d) and returns an array of shape (..., 1).
    :param noise_std: float.
    :return: Callable that takes in an array of shape (..., d) and returns an array of shape (..., 1).
    """
    return lambda x: obj_func(x) + noise_std * torch.randn(size=x.shape[:-1] + (1,))
