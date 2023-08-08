from botorch import test_functions
import gpflow as gpf
import numpy as np
import pickle
import tensorflow as tf
import tensorflow_probability as tfp
import torch

from core.utils import uniform_samples


def get_objective(kernel, bounds, config):
    task = config.task

    if task == "gp":
        obj_func = sample_gp_prior(
            kernel=kernel,
            bounds=bounds,
            num_points=config.gp_sample_num_points,
            jitter=config.jitter,
        )
    elif task == "hartmann":
        neg_obj = test_functions.Hartmann(dim=3, negate=True)
        orig_bounds = neg_obj.bounds.to(dtype=torch.double)
        unsqueezed_obj = lambda x: neg_obj(x).unsqueeze(-1)
        obj_func = input_transform_wrapper(obj_func=unsqueezed_obj, bounds=orig_bounds)

    elif task == "plant":
        NH3pH_leaf_max_area_func, _, _ = create_synth_funcs(params="NH3pH")

        def NH3pH_wrapper(vals):
            X = np.zeros(vals.shape)
            X[:, 0] = vals[:, 1] * 30000
            X[:, 1] = 2.5 + vals[:, 0] * (6.5 - 2.5)

            mean, _ = NH3pH_leaf_max_area_func(X)
            leaf_mean = 67.2466342112483
            leaf_std = 59.347376136036964
            return torch.tensor((mean - leaf_mean) / leaf_std)

        obj_func = NH3pH_wrapper
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


def create_synth_funcs(params):
    """

    :param params:
    :return:
    """
    gp_leaf_dict = pickle.load(open(f"data/plant/{params}_gp_leaf_dict.p", "rb"))
    (
        leaf_mean,
        leaf_std,
        tbm_mean,
        tbm_std,
        tbs_mean,
        tbs_std,
        num_inducing,
        d,
    ) = pickle.load(open(f"data/plant/{params}_req_variables.p", "rb"))

    gp_leaf = init_heteroscedastic_gp(num_inducing=num_inducing, d=d)
    gpf.utilities.multiple_assign(gp_leaf, gp_leaf_dict)

    def leaf_max_area_func(X):
        """
        Returns the predictive mean and variance of the maximum leaf area.
        :param X: Array of shape (num_preds, d).
        :return: Tuple (array of shape (num_preds, 1), array of shape (num_preds, 1). First element is mean, second
        is variance.
        """
        mean, var = gp_leaf.predict_y(X)
        return mean.numpy() * leaf_std + leaf_mean, var.numpy() * (leaf_std**2)

    return leaf_max_area_func, None, None


def init_heteroscedastic_gp(num_inducing, d):
    """
    Initializes default heteroscedastic GP, so that we can load the parameters later.
    :param num_inducing:
    :param d:
    :return:
    """
    likelihood = gpf.likelihoods.HeteroskedasticTFPConditional(
        distribution_class=tfp.distributions.Normal,  # Gaussian Likelihood
        scale_transform=tfp.bijectors.Exp(),  # Exponential Transform
    )

    kernels = [
        gpf.kernels.SquaredExponential(lengthscales=np.ones(d)),
        gpf.kernels.SquaredExponential(lengthscales=np.ones(d)),
    ]

    kernel = gpf.kernels.SeparateIndependent(kernels)

    Z = tf.zeros((num_inducing, d))
    inducing_variable = gpf.inducing_variables.SeparateIndependentInducingVariables(
        [
            gpf.inducing_variables.InducingPoints(Z),  # This is U1 = f1(Z1)
            gpf.inducing_variables.InducingPoints(Z),  # This is U2 = f2(Z2)
        ]
    )

    model = gpf.models.SVGP(
        kernel=kernel,
        likelihood=likelihood,
        inducing_variable=inducing_variable,
        num_latent_gps=likelihood.latent_dim,
    )
    return model
