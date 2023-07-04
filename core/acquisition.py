import torch

from core.utils import cross_product


def acquire(gp, decision_points, context_points, config):
    acquisition = config.acquisition
    joint_points = cross_product(decision_points, context_points)

    if acquisition == "ts":
        return thompson_sampling(gp=gp, points=joint_points)
    else:
        raise NotImplementedError


def thompson_sampling(gp, points):
    # TODO: change to stochastic version
    gp.eval()
    pred = gp(points)
    sample = pred.sample()  # (len(points), )
    best_idx = torch.argmax(sample)

    return points[best_idx][None, :]
