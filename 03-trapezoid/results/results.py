import json
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

# Load data
with open("results.json", "r") as f:
    data = json.load(f)

# Implementations to track
implementations = [
    # "CPU sequential",
    # "CPU multithread",
    "GPU naive",
    "GPU shared memory tree-structured sum",
    "GPU warp shuffle tree-structured sum",
    "GPU shared memory dissemination sum",
    "GPU warp shuffle dissemination sum",
]

# Structure:
# results[n][blockSize][implementation] = list of times
results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

# Parse data
for key, runs in data.items():
    for run in runs:
        n = run["n"]
        block_size = run["blockSize"]

        for impl in implementations:
            time = run[impl]["time (ms)"]
            results[n][block_size][impl].append(time)

# Compute averages
avg_results = defaultdict(lambda: defaultdict(dict))

for n in results:
    for block_size in results[n]:
        for impl in implementations:
            times = results[n][block_size][impl]
            avg_results[n][block_size][impl] = np.mean(times)

# Plot
for n in sorted(avg_results.keys()):
    plt.figure(figsize=(10, 6))

    block_sizes = sorted(avg_results[n].keys())

    for impl in implementations:
        y = [avg_results[n][bs][impl] for bs in block_sizes]
        plt.plot(block_sizes, y, marker='o', label=impl)

    plt.title(f"Average Time vs Block Size (n = {n})")
    plt.xlabel("Block Size")
    plt.ylabel("Time (ms)")
    # plt.yscale("log")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"full_plot_n_{n}.png")