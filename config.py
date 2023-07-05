import argparse
from pathlib import Path


def get_config(add_compulsory_args):
    parser = argparse.ArgumentParser()

    if add_compulsory_args:
        parser.add_argument("task", type=str)
        parser.add_argument("distance_name", type=str)
        parser.add_argument("alpha", type=float)
        parser.add_argument("eps_1", type=float)
        parser.add_argument("eps_2", type=float)
        parser.add_argument("acquisition", type=str)
        parser.add_argument("seed", type=int)

    parser.add_argument("--decision_dims", type=int, default=2)
    parser.add_argument("--context_dims", type=int, default=1)
    parser.add_argument("--decision_density_per_dim", type=int, default=32)
    parser.add_argument("--context_density_per_dim", type=int, default=16)

    parser.add_argument("--T", type=int, default=400)
    parser.add_argument("--outputscale", type=float, default=1.0)
    parser.add_argument("--lengthscale", type=float, default=0.1)
    parser.add_argument("--noise_std", type=float, default=0.01)
    parser.add_argument("--num_init_points", type=int, default=5)

    parser.add_argument("--gp_sample_num_points", type=int, default=1000)

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

    filename = (
        f"{task}_dist{config.distance_name}_{config.alpha}_{config.eps_1}"
        f"_{config.eps_2}_acq{config.acquisition}_seed{config.seed}"
    )
    config.filename = filename.replace(".", ",")

    return config
