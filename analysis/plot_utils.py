import matplotlib.pyplot as plt
import numpy as np

def plot_speedup(df_list, title, label, color='blue', plot_ideal=True):
    # Compute minibatch time for all DataFrames
    for df in df_list:
        df["minibatch_time"] = df["forward_time"] + df["backward_time"]

    # Base time from the single-worker (1 GPU) run
    base_time = df_list[0]["minibatch_time"].sum()

    # Degree of parallelism: 1, 2, 4, 8, ...
    x_values = [2**i for i in range(len(df_list))]

    # Speedup = T(1) / T(n)
    speedups = [base_time / df["minibatch_time"].sum() for df in df_list]

    # Plot measured speedup
    plt.plot(x_values, speedups, marker='o', label=label, color=color)

    # Plot ideal linear speedup
    if plot_ideal:
        ideal_speedup = x_values  # i.e., S(n) = n
        plt.plot(x_values, ideal_speedup, linestyle='--', color='gray', label="Ideal Speedup")

    plt.xticks(x_values, [f"{n}" for n in x_values])
    plt.xlabel("Number of Processes (GPUs)")
    plt.ylabel("Speedup T(1)/T(n)")
    plt.title(title)
    plt.legend()
    plt.grid(True)

def plot_memory_usage(memory_data, title): #TODO: use scientific notation for y-axis
    """
    memory_data: dict of format {
        minibatch_size: {
            ddp_degree: df
        }
    }
    Example:
    {
        64: {2: df_2_64, 4: df_4_64},
        128: {2: df_2_128, 4: df_4_128, 8: df_8_128},
        256: {2: df_2_256, 4: df_4_256}
    }
    """

    minibatches = sorted(memory_data.keys())
    all_ddp_degrees = sorted({ddp for ddp_dict in memory_data.values() for ddp in ddp_dict})
    n_minibatches = len(minibatches)
    n_ddp = len(all_ddp_degrees)

    x = np.arange(n_minibatches)
    bar_width = 0.8 / n_ddp  # shrink bars to fit all groups

    _, ax = plt.subplots(figsize=(10, 6))

    for i, ddp in enumerate(all_ddp_degrees):
        means = []
        for mb in minibatches:
            if ddp in memory_data[mb]:
                mean = memory_data[mb][ddp]["peak_memory_usage(MB)"].mean()
            else:
                mean = np.nan  # skip if missing
            means.append(mean)

        offset = (i - (n_ddp - 1) / 2) * bar_width
        ax.bar(x + offset, means, width=bar_width, label=f"Parallelism {ddp}")

    ax.set_xlabel("Minibatch Size")
    ax.set_ylabel("Peak Memory Usage (MB)")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([str(mb) for mb in minibatches])
    ax.legend()
    plt.tight_layout()
    plt.show()