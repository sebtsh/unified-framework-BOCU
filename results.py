import matplotlib.pyplot as plt
import numpy as np
import os.path
from pathlib import Path
import pickle

from config import get_alpha_beta


tasks = ["gp", "hartmann", "plant", "infection"]
distance_names = ["tv", "mmd"]
unc_objs = ["dro", "wcs", "gen"]
acquisitions = ["ts", "ucb", "ucbu", "random", "so", "ro"]
seeds = range(10)

text_size = 14
tick_size = 10
dpi = 300

color_dict = {
    "random": "black",
    "so": "#f76c5e",  # red
    "ro": "#F6B678",  # yellow
    "ucb": "#B9B494",  # brown
    "ucbu": "#7BB1B0",  # green
    "ts": "#00ABE7",  # blue
}
acq_name_dict = {
    "random": "Random",
    "ucb": "UCB-1",
    "ucbu": "UCB-2",
    "ts": "TS",
    "so": "SO",
    "ro": "RO",
}

save_dir = "results/summary_results/"
Path(save_dir).mkdir(parents=True, exist_ok=True)
fig_cumu, all_axs_cumu = plt.subplots(
    len(tasks), len(unc_objs) * len(distance_names), figsize=(16, 20)
)

for i, task in enumerate(tasks):
    print(f"================ {task} ================")

    base_dir = "results/" + task + "/"
    pickles_dir = base_dir + "pickles/"

    for j, unc_obj in enumerate(unc_objs):
        alpha, beta = get_alpha_beta(unc_obj)

        for k, distance_name in enumerate(distance_names):
            axs_cumu = all_axs_cumu[i][j * 2 + k]
            axs_cumu.grid(which="major")
            axs_cumu.grid(which="minor", linestyle=":", alpha=0.3)

            for acquisition in acquisitions:
                if unc_obj == "dro":  # special rules for dro
                    if acquisition == "ucbu":  # in dro, ucbu is equivalent to ucb
                        continue

                color = color_dict[acquisition]
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
                    cumu_regrets.append(cumu_regret)

                cumu_regrets = np.array(cumu_regrets)

                mean_cumu_regrets = np.mean(cumu_regrets, axis=0)
                std_err_cumu_regrets = np.std(cumu_regrets, axis=0) / np.sqrt(
                    len(cumu_regrets)
                )

                T = len(mean_cumu_regrets)
                xaxis = np.arange(T)

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

                # axs_cumu.set_xlabel("Iteration $t$", size=text_size)
                # axs_cumu.set_ylabel("Cumulative regret", size=text_size)
                axs_cumu.tick_params(labelsize=tick_size)
                # axs_cumu.legend(fontsize=text_size - 2)

    fig_cumu.tight_layout()
    fig_cumu.subplots_adjust(hspace=0.15)
    fig_cumu.savefig(
        save_dir + f"cumu_regret.png",
        dpi=dpi,
        bbox_inches="tight",
        format="png",
    )
