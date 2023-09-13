# A Unified Framework for Bayesian Optimization under Contextual Uncertainty

This code repository accompanies the paper "A Unified Framework for Bayesian Optimization under 
Contextual Uncertainty".


## Requirements
The Python packages used in the experiments are listed in `environment.yml`, from which a fresh
conda environment can be created.

## Running experiments
To run all experiments, simply run `job_creator.py` to create a list of jobs at `jobs/jobs.txt`,
then run `job_runner.py`to run them sequentially. Alternatively, to run a specific experiment,
```
python exp.py [task] [distance_name] [unc_obj] [acquisition] [seed]
```
where the valid strings for each argument can be found at the top of `job_creator.py`.