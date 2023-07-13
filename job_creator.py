import os.path
from pathlib import Path
import shutil

create_jobs = True
is_delete = True

tasks = ["gp"]
distance_names = ["mmd", "tv"]
unc_objs = ["dro, wcs, gen"]
alphas = [1.0]
betas = [1.0]
acquisitions = ["ts", "ucb", "random"]
seeds = range(5)


missing_filenames = []
counter = 1
for task in tasks:
    base_dir = "results/" + task + "/"
    pickles_dir = base_dir + "pickles/"
    Path(pickles_dir).mkdir(parents=True, exist_ok=True)
    for distance_name in distance_names:
        for unc_obj in unc_objs:
            for acquisition in acquisitions:
                for seed in seeds:
                    filename = (
                        f"{task}_dist{distance_name}_unc{unc_obj}"
                        f"_acq{acquisition}_seed{seed}"
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
