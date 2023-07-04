import datetime
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.likelihoods.gaussian_likelihood import GaussianLikelihood
import numpy as np
import torch


def get_discrete_fvals(fvals, decision_points, context_points):
    """
    Reshapes fvals into a 2D array res such that the context values of the decision point at index i is res[i].
    WARNING: Assumes that fvals is the result of a function applied to cross_product(decision_points, context_points).
    :param fvals: array of shape (|dec| * |con|, ).
    :param decision_points:
    :param context_points:
    :return:
    """
    return torch.reshape(fvals, (len(decision_points), len(context_points)))


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


def get_index_of_1d_array_in_2d_array(one_arr, two_arr):
    for i in range(len(two_arr)):
        if np.allclose(one_arr, two_arr[i]):
            return i

    raise ValueError


def get_indices_from_ref_array(input, ref):
    indices = []
    for i in range(len(input)):
        index = get_index_of_1d_array_in_2d_array(input[i], ref)
        indices.append(index)
    return np.array(indices)
