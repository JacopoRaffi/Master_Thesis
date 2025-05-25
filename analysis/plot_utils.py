import matplotlib.pyplot as plt

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