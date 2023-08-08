import matplotlib.pyplot as plt
import numpy as np
import os.path
from pathlib import Path
import pickle

from config import get_alpha_beta


tasks = ["gp", "plant", "infection"]
distance_names = ["mmd", "tv"]
unc_objs = ["dro", "wcs", "gen"]
acquisitions = ["ts", "ucb", "random"]
seeds = range(5)

text_size = 16
tick_size = 10
dpi = 200

color_dict = {
    "random": "black",
    "ucb": "#d7263d",
    "ts": "#00a6ed",
}
acq_name_dict = {
    "random": "Random",
    "ucb": "UCB",
    "ts": "TS",
}

for task in tasks:
    print(f"================ {task} ================")

    base_dir = "results/" + task + "/"
    save_dir = "results/summary_results/"
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    pickles_dir = base_dir + "pickles/"

    for unc_obj in unc_objs:
        alpha, beta = get_alpha_beta(unc_obj)

        fig_simple, all_axs_simple = plt.subplots(1, len(distance_names))
        fig_cumu, all_axs_cumu = plt.subplots(1, len(distance_names))

        for i, distance_name in enumerate(distance_names):
            axs_simple = all_axs_simple[i]
            axs_cumu = all_axs_cumu[i]
            axs_simple.grid(which="major")
            axs_simple.grid(which="minor", linestyle=":", alpha=0.3)
            axs_cumu.grid(which="major")
            axs_cumu.grid(which="minor", linestyle=":", alpha=0.3)

            for acquisition in acquisitions:
                color = color_dict[acquisition]
                simple_regrets = []
                cumu_regrets = []
                for seed in seeds:
                    filename = (
                        f"{task}_dist{distance_name}_unc{unc_obj}"
                        f"_acq{acquisition}_seed{seed}"
                    )
                    filename = filename.replace(".", ",") + ".p"
                    if not os.path.isfile(pickles_dir + filename):
                        print(f"{filename} is missing")
                        continue

                    _, _, simple_regret, cumu_regret = pickle.load(
                        open(pickles_dir + filename, "rb")
                    )
                    simple_regrets.append(simple_regret)
                    cumu_regrets.append(cumu_regret)

                simple_regrets = np.array(simple_regrets)
                cumu_regrets = np.array(cumu_regrets)

                mean_simple_regrets = np.mean(simple_regrets, axis=0)
                std_err_simple_regrets = np.std(simple_regrets, axis=0) / np.sqrt(
                    len(simple_regrets)
                )

                mean_cumu_regrets = np.mean(cumu_regrets, axis=0)
                std_err_cumu_regrets = np.std(cumu_regrets, axis=0) / np.sqrt(
                    len(cumu_regrets)
                )

                T = len(mean_simple_regrets)
                xaxis = np.arange(T)

                axs_simple.plot(
                    xaxis,
                    mean_simple_regrets,
                    label=acq_name_dict[acquisition],
                    color=color,
                )
                axs_simple.fill_between(
                    xaxis,
                    mean_simple_regrets - std_err_simple_regrets,
                    mean_simple_regrets + std_err_simple_regrets,
                    alpha=0.2,
                    color=color,
                )

                axs_cumu.plot(
                    xaxis,
                    mean_cumu_regrets,
                    label=acq_name_dict[acquisition],
                    color=color,
                )
                axs_cumu.fill_between(
                    xaxis,
                    mean_cumu_regrets - std_err_cumu_regrets,
                    mean_cumu_regrets + std_err_cumu_regrets,
                    alpha=0.2,
                    color=color,
                )

                axs_simple.set_title(f"{distance_name}", size=text_size)
                axs_simple.set_xlabel("Iteration $t$", size=text_size)
                axs_simple.set_ylabel("Simple regret", size=text_size)
                axs_simple.tick_params(labelsize=tick_size)
                axs_simple.legend(fontsize=text_size - 2)
                axs_simple.set_yscale("log")

                axs_cumu.set_title(f"{distance_name}", size=text_size)
                axs_cumu.set_xlabel("Iteration $t$", size=text_size)
                axs_cumu.set_ylabel("Cumulative regret", size=text_size)
                axs_cumu.tick_params(labelsize=tick_size)
                axs_cumu.legend(fontsize=text_size - 2)

        fig_simple.suptitle(
            f"{task}: $\\alpha={alpha}$, $\\epsilon_2={beta}$",
            size=text_size,
        )
        fig_simple.tight_layout()
        fig_simple.savefig(
            save_dir + f"{task}-{unc_obj}-simple_regret.pdf",
            dpi=dpi,
            bbox_inches="tight",
            format="pdf",
        )

        fig_cumu.suptitle(
            f"{task}: $\\alpha={alpha}$, $\\epsilon_2={beta}$",
            size=text_size,
        )
        fig_cumu.tight_layout()
        fig_cumu.savefig(
            save_dir + f"{task}-{unc_obj}_cumu_regret.png",
            dpi=dpi,
            bbox_inches="tight",
            format="png",
        )
