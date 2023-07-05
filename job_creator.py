import os.path
from pathlib import Path
import shutil

create_jobs = True
is_delete = True

tasks = ["gp"]
distance_names = ["mmd", "tv"]
alphas = [1.0]
eps_1s = [0.5]
eps_2s = [1.0]
acquisitions = ["ts", "ucb", "random"]
seeds = range(5)


missing_filenames = []
counter = 1
for task in tasks:
    base_dir = "results/" + task + "/"
    pickles_dir = base_dir + "pickles/"
    Path(pickles_dir).mkdir(parents=True, exist_ok=True)
    for distance_name in distance_names:
        for alpha in alphas:
            for eps_1 in eps_1s:
                for eps_2 in eps_2s:
                    for acquisition in acquisitions:
                        for seed in seeds:
                            filename = (
                                f"{task}_dist{distance_name}_{alpha}_{eps_1}"
                                f"_{eps_2}_acq{acquisition}_seed{seed}"
                            )
                            filename = filename.replace(".", ",") + ".p"
                            if not os.path.isfile(pickles_dir + filename):
                                if filename not in missing_filenames:
                                    missing_filenames.append(filename)
                                    print(f"{counter}. {filename} is missing")
                                    counter += 1

if create_jobs:
    # Create job files
    job_dir = "jobs/"
    if os.path.exists(job_dir):  # empty the job_dir directory
        shutil.rmtree(job_dir)
    Path(job_dir).mkdir(parents=True, exist_ok=True)

    for i, f in enumerate(missing_filenames):
        with open(job_dir + f"job.txt", "a") as file:
            file.write(f"{f}\n")
    open(job_dir + f"job.txt.lock", "a")
