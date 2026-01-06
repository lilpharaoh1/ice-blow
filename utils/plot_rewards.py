import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def compute_timesteps(data):
    timesteps = []
    t = 0
    for d in data:
        timesteps.append(t)
        t += d["length"]
    return np.array(timesteps)


def moving_average(x, window):
    if len(x) < window:
        return np.array([])
    return np.convolve(x, np.ones(window) / window, mode="valid")


def ewma(x, alpha):
    y = np.zeros_like(x, dtype=np.float32)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1 - alpha) * y[i - 1]
    return y

def load_metrics(path):
    with open(path, "r") as f:
        return json.load(f)


def collect_runs(root_dir, eval_only=False):
    """
    Walks root_dir and finds all metrics.json files.
    Groups them by run directory.
    """
    runs = {}

    for root, dirs, files in os.walk(root_dir):
        if "metrics.json" in files:
            metrics_path = os.path.join(root, "metrics.json")
            data = load_metrics(metrics_path)

            if eval_only:
                data = [d for d in data if d.get("eval", False)]
            else:
                data = [d for d in data if not d.get("eval", False)]

            if len(data) == 0:
                continue

            runs[root] = data

    return runs

def plot_single_run(
    data,
    label=None,
    smooth="ewma",
    window=50,
    alpha=0.1,
):
    timesteps = compute_timesteps(data)
    rewards = np.array([d["reward"] for d in data])

    # Raw rewards (dots)
    plt.scatter(
        timesteps,
        rewards,
        s=10,
        alpha=0.4,
        label=f"{label} (raw)" if label else None,
    )

    # Smoothed curve
    if smooth == "ma":
        smoothed = moving_average(rewards, window)
        smoothed_ts = timesteps[window - 1 :]
    elif smooth == "ewma":
        smoothed = ewma(rewards, alpha)
        smoothed_ts = timesteps
    else:
        raise ValueError(f"Unknown smoothing method: {smooth}")

    plt.plot(
        smoothed_ts,
        smoothed,
        linewidth=2.0,
        label=f"{label} ({smooth})" if label else None,
    )



def plot_mean_std(
    runs,
    smooth="ewma",
    window=50,
    alpha=0.1,
    bin_size=1000,
):
    """
    Aggregates runs by timestep bins.
    """

    bins = defaultdict(list)

    for data in runs.values():
        timesteps = compute_timesteps(data)
        rewards = np.array([d["reward"] for d in data])

        for t, r in zip(timesteps, rewards):
            b = int(t // bin_size) * bin_size
            bins[b].append(r)

    xs = np.array(sorted(bins.keys()))
    means = np.array([np.mean(bins[x]) for x in xs])
    stds = np.array([np.std(bins[x]) for x in xs])

    # Raw mean dots
    plt.scatter(
        xs,
        means,
        s=15,
        alpha=0.6,
        label="mean reward (raw)",
    )

    # Smoothed mean
    if smooth == "ma":
        smoothed = moving_average(means, window)
        smoothed_x = xs[window - 1 :]
    elif smooth == "ewma":
        smoothed = ewma(means, alpha)
        smoothed_x = xs
    else:
        raise ValueError(f"Unknown smoothing method: {smooth}")

    plt.plot(
        smoothed_x,
        smoothed,
        linewidth=2.5,
        label=f"mean reward ({smooth})",
    )

    # Std band
    plt.fill_between(
        xs,
        means - stds,
        means + stds,
        alpha=0.2,
        label="std",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, required=True,
                        help="Root directory containing run subfolders")
    parser.add_argument("--aggregate", action="store_true",
                        help="Aggregate multiple runs (mean/std)")
    parser.add_argument("--eval", action="store_true",
                        help="Plot evaluation episodes instead of training")
    parser.add_argument("--title", type=str, default="Episode Reward")
    parser.add_argument("--save", type=str, default=None,
                        help="Path to save figure (png)")
    parser.add_argument("--smooth", type=str, default="ewma",
                        choices=["ewma", "ma"],
                        help="Smoothing method")
    parser.add_argument("--window", type=int, default=50,
                        help="Window size for moving average")
    parser.add_argument("--alpha", type=float, default=0.1,
                        help="EWMA smoothing factor")
    parser.add_argument(
        "--bin-size",
        type=int,
        default=1000,
        help="Timestep bin size for aggregation",
    )

    args = parser.parse_args()

    runs = collect_runs(args.logdir, eval_only=args.eval)

    if len(runs) == 0:
        raise RuntimeError("No metrics.json files found")

    plt.figure(figsize=(8, 5))

    if args.aggregate:
        plot_mean_std(runs)
    else:
        for run_name, data in runs.items():
            label = os.path.basename(run_name)
            plot_single_run(data, label=label)

    plt.xlabel("Environment Timesteps")
    plt.ylabel("Reward")
    plt.title(args.title)
    plt.legend()
    plt.grid(True)

    if args.save is not None:
        plt.savefig(args.save, dpi=150, bbox_inches="tight")
    else:
        plt.show()


if __name__ == "__main__":
    main()
