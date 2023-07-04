import datetime
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.likelihoods.gaussian_likelihood import GaussianLikelihood
import torch


def construct_grid_1d(min_range, max_range, grid_density):
    return torch.linspace(min_range, max_range, grid_density)[:, None]


def construct_grid(bounds, density_per_dim):
    """

    :param bounds: array of shape (2, d).
    :param density_per_dim: int.
    :return:
    """
    lowers = bounds[0]
    uppers = bounds[1]
    d = len(lowers)
    decision_points = construct_grid_1d(lowers[0], uppers[0], density_per_dim)
    for i in range(d - 1):
        decision_points = cross_product(
            decision_points,
            construct_grid_1d(lowers[i + 1], uppers[i + 1], density_per_dim),
        )

    return decision_points


def cross_product(x, y):
    """

    :param x: array of shape (m, d_x)
    :param y: array of shape (n, d_y)
    :return:  array of shape (m * n, d_x + d_y)
    """
    m, d_x = x.shape
    n, d_y = y.shape
    x_temp = torch.tile(x[:, :, None], (1, n, 1))
    x_temp = torch.reshape(x_temp, [m * n, d_x])
    y_temp = torch.tile(y, (m, 1))
    return torch.cat([x_temp, y_temp], dim=-1)


def construct_bounds(lower, upper, d):
    return torch.tensor([[lower] * d, [upper] * d])


def log(msg):
    print(str(datetime.datetime.now()) + " - " + msg)


def uniform_samples(bounds, n_samples):
    low = bounds[0]
    high = bounds[1]
    d = len(low)
    return torch.rand(size=(n_samples, d), dtype=torch.double) * (high - low) + low


def create_kernel(config):
    dims = config.decision_dims + config.context_dims
    kernel = ScaleKernel(RBFKernel(ard_num_dims=dims))
    kernel.outputscale = torch.tensor(config.outputscale)
    kernel.base_kernel.lengthscale = torch.tensor([config.lengthscale] * dims)

    return kernel


def create_likelihood(config):
    likelihood = GaussianLikelihood()
    likelihood.noise = torch.tensor(config.noise_std**2)

    return likelihood
