import matplotlib.pyplot as plt
import numpy as np

def plot_speedup(df_list, title, label, color='blue', plot_ideal=True):
    for df in df_list:
        df["minibatch_time"] = df["forward_time"] + df["backward_time"]

    base = df_list[0]["minibatch_time"].sum()
    x_values = [2**i for i in range(1, len(df_list) + 1)]

    speedups = [base / df["minibatch_time"].sum() for df in df_list]

    #plt.figure(figsize=(10, 5))
    plt.plot(x_values, speedups, marker='o', label=label, color=color)
    
    if plot_ideal:
        ideal_speedup = [2**i for i in range(0, len(df_list))]
        plt.plot(x_values, ideal_speedup, linestyle='--', color='gray', label="Ideal Speedup")
    
    plt.xticks(x_values, [f"{ddp}" for ddp in x_values])
    plt.xlabel("Degree of Parallelism")
    plt.ylabel("Normalized Speedup")
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