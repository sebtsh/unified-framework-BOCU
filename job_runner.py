from filelock import FileLock

from config import get_config
from exp import run_exp

job_dir = "jobs/"
job_filename = job_dir + f"job.txt"
job_lockname = job_dir + f"job.txt.lock"


def parse_params(param_string):
    """
    WARNING: assumes seeds have only 1 digit!
    :param job_file:
    :return:
    """
    params = param_string.split(sep="_")
    dic = {
        "task": params[0],
        "distance_name": params[1][4:],
        "alpha": float(params[2].replace(",", ".")),
        "eps_1": float(params[3].replace(",", ".")),
        "eps_2": float(params[4].replace(",", ".")),
        "acquisition": params[5][3:],
        "seed": int(params[6][4]),
    }

    return dic


while True:
    lock = FileLock(job_lockname)
    with lock:
        with open(job_filename, "r") as fin:
            data = fin.read().splitlines(True)
            if len(data) == 0:
                print("job.txt is empty, exiting")
                break

        with open(job_filename, "w") as fout:
            fout.writelines(data[1:])

    param_string = data[0]
    param_dict = parse_params(param_string)
    print(f"params: {param_dict}")

    task = param_dict["task"]

    config = get_config(add_compulsory_args=False)
    for k in param_dict.keys():
        setattr(config, k, param_dict[k])

    run_exp(config)
