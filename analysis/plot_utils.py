import matplotlib.pyplot as plt
import numpy as np

def plot_speedup(df_list, title, label, color="blue", linestyle="-", plot_ideal=True):
    # Compute minibatch time for all DataFrames
    for df in df_list:
        df["minibatch_time"] = df["forward_time"] + df["backward_time"]

    # Base time from the single-worker (1 GPU) run
    base_time = df_list[0]["minibatch_time"].mean()

    # Degree of parallelism: 1, 2, 4, 8, ...
    x_values = [2**i for i in range(len(df_list))]

    # Speedup = T(1) / T(n)
    speedups = [base_time / df["minibatch_time"].mean() for df in df_list]

    # Plot measured speedup
    plt.plot(x_values, speedups, marker='o', label=label, color=color, linestyle=linestyle)

    # Plot ideal linear speedup
    if plot_ideal:
        ideal_speedup = x_values  # i.e., S(n) = n
        plt.plot(x_values, ideal_speedup, linestyle='--', color='gray', label="Ideal Scalability")
    
    plt.xticks(x_values, [f"{ddp}" for ddp in x_values])
    plt.xlabel("Degree of Parallelism")
    plt.ylabel("Scalability")
    plt.title(title)
    plt.legend()
    plt.grid(True)

def plot_memory_usage(memory_data, title, to_gb=True):
    """
    memory_data: dict in format {
        minibatch_size: {
            ddp_degree: peak_memory_usage_in_MB (single value)
        }
    }
    Example:
    {
        64: {1: 20000, 2: 10000, 4: 4000},
        128: {1: 25000, 2: 13000, 4: 6000, 8: 3500}
        256: {1: 30000, 2: 15000, 4: 8000, 8: 5000}
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
        values = []
        for mb in minibatches:
            if ddp in memory_data[mb]:
                val = memory_data[mb][ddp]
                if to_gb:
                    val = val / 1024  # convert MB to GB
            else:
                val = np.nan
            values.append(val)

        offset = (i - (n_ddp - 1) / 2) * bar_width
        ax.bar(x + offset, values, width=bar_width, label=f"Parallelism {ddp}")

    ax.set_xlabel("Minibatch Size")
    ylabel = "Peak Memory Usage (GB)" if to_gb else "Peak Memory Usage (MB)"
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([str(mb) for mb in minibatches])
    ax.legend()

    plt.tight_layout()