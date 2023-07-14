import argparse
from pathlib import Path


def get_config(add_compulsory_args):
    parser = argparse.ArgumentParser()

    if add_compulsory_args:
        parser.add_argument("task", type=str)
        parser.add_argument("distance_name", type=str)
        parser.add_argument("unc_obj", type=str)
        parser.add_argument("acquisition", type=str)
        parser.add_argument("seed", type=int)

    parser.add_argument("--ref_mean", type=float, default=0.5)
    parser.add_argument("--ref_var", type=float, default=0.2)

    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument("--outputscale", type=float, default=1.0)
    parser.add_argument("--lengthscale", type=float, default=0.1)
    parser.add_argument("--noise_std", type=float, default=0.01)
    parser.add_argument("--num_init_points", type=int, default=5)

    parser.add_argument("--kernel", type=str, default="se")
    parser.add_argument("--gp_sample_num_points", type=int, default=1000)
    parser.add_argument("--rff_num_samples", type=int, default=1024)
    parser.add_argument("--finite_diff_h", type=float, default=0.001)
    parser.add_argument("--beta", type=float, default=2.0, help="beta for GP-UCB")
    parser.add_argument("--jitter", type=float, default=1e-06)

    config = parser.parse_args()

    return config


def set_dir_attributes(config):
    task = config.task
    base_dir = "results/" + task + "/"
    pickles_save_dir = base_dir + "pickles/"
    figures_save_dir = base_dir + "figures/"
    Path(pickles_save_dir).mkdir(parents=True, exist_ok=True)
    Path(figures_save_dir).mkdir(parents=True, exist_ok=True)
    config.pickles_save_dir = pickles_save_dir
    config.figures_save_dir = figures_save_dir

    filename = f"{task}_dist{config.distance_name}_unc{config.unc_obj}_acq{config.acquisition}_seed{config.seed}"
    config.filename = filename.replace(".", ",")

    return config


def set_unc_attributes(config):
    unc_obj = config.unc_obj
    if unc_obj == "dro":
        alpha = 1.0
        beta = 0.0
    elif unc_obj == "wcs":
        alpha = 0.0
        beta = 1.0
    elif unc_obj == "gen":
        alpha = 1.0
        beta = 1.0
    else:
        raise NotImplementedError

    config.alpha = alpha
    config.beta = beta

    return config


def set_task_attributes(config):
    task = config.task

    if task == "gp":
        decision_dims = 2
        context_dims = 2
        decision_density_per_dim = 32
        context_density_per_dim = 8
    elif task == "plant":
        decision_dims = 3
        context_dims = 2
        decision_density_per_dim = 10
        context_density_per_dim = 8
    elif task == "infection":
        decision_dims = 2
        context_dims = 3
        decision_density_per_dim = 32
        context_density_per_dim = 4
    else:
        raise NotImplementedError

    config.decision_dims = decision_dims
    config.context_dims = context_dims
    config.decision_density_per_dim = decision_density_per_dim
    config.context_density_per_dim = context_density_per_dim

    return config
