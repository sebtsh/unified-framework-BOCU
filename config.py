import argparse
from pathlib import Path
import torch


def get_config(add_compulsory_args):
    parser = argparse.ArgumentParser()

    if add_compulsory_args:
        parser.add_argument("task", type=str)
        parser.add_argument("acquisition", type=str)
        parser.add_argument("distance_name", type=str)
        parser.add_argument("alpha", type=float)
        parser.add_argument("eps_1", type=float)
        parser.add_argument("eps_2", type=float)
        parser.add_argument("seed", type=int)

    parser.add_argument("--decision_dims", type=int, default=2)
    parser.add_argument("--context_dims", type=int, default=2)
    parser.add_argument("--decision_density_per_dim", type=int, default=10)
    parser.add_argument("--context_density_per_dim", type=int, default=8)

    parser.add_argument("--T", type=int, default=200)
    parser.add_argument("--outputscale", type=float, default=1.0)
    parser.add_argument("--lengthscale", type=float, default=0.2)
    parser.add_argument("--noise_std", type=float, default=0.01)
    parser.add_argument("--num_init_points", type=int, default=5)

    parser.add_argument("--gp_sample_num_points", type=int, default=1000)

    parser.add_argument("--finite_diff_h", type=float, default=0.001)

    config = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config.device = device

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

    filename = f"{task}_acq{config.acquisition}_seed{config.seed}"
    config.filename = filename.replace(".", ",")

    return config
