import yaml
from argparse import Namespace
from copy import deepcopy


def setup_run_list(run_file):
    with open(run_file, "r") as f:
        cfg = yaml.safe_load(f)

    n_seeds = cfg.get("n_seeds", 1)
    start_seed = cfg.get("start_seed", 0)

    defaults = cfg.get("defaults", {})
    runs = cfg["runs"]

    run_list = []

    for seed in range(start_seed, start_seed + n_seeds):
        for run_cfg in runs:
            run = deepcopy(run_cfg)

            # Merge defaults
            for k, v in defaults.items():
                if k not in run:
                    run[k] = deepcopy(v)
                elif isinstance(v, dict):
                    for kk, vv in v.items():
                        run[k].setdefault(kk, vv)

            run["seed"] = seed
            run["experiment_name"] = cfg["experiment_name"]

            # Generate run name
            run["run_name"] = f"{run['name']}_seed{seed}"
            run["output_path"] = f"results/{cfg['experiment_name']}/{run['run_name']}"

            run_list.append(Namespace(**run))

    return run_list
