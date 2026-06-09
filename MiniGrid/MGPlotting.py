from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def aggregate_by_iteration(callback):
    data = defaultdict(list)

    for i, r in zip(callback.iterations, callback.rewards):
        data[i].append(r)

    x = np.array(sorted(data.keys()))
    y = np.array([np.mean(data[i]) for i in x])

    return x, y


def moving_average(y, window=10):
    return pd.Series(y).rolling(window=window, min_periods=1).mean().to_numpy()


def plot_learning_curve(callback):
    x, y = aggregate_by_iteration(callback)

    # Downsample to reduce clutter
    x = x[::5]
    y = y[::5]

    # Smooth curve
    window = 50
    if len(y) >= window:
        y_smooth = moving_average(y, window)
        x_smooth = x[: len(y_smooth)]
    else:
        x_smooth, y_smooth = x, y

    plt.figure(figsize=(10, 5))

    # Plot curve
    plt.plot(x_smooth, y_smooth, linewidth=2, label="Average Reward")

    plt.xlabel("Training Iterations")
    plt.ylabel("Average Episode Reward")
    plt.title("Learning Progress of PPO Agent in MiniGrid Environment")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.savefig("Data/Figures/MiniGrid_RL_Learning.png", dpi=300, bbox_inches="tight")

    plt.show()


def plot_metrics(avg_post_rewards, avg_pre_rewards, avg_agreements, avg_entropies, window=10):
    generations = np.array(avg_pre_rewards.index)

    pre_rewards = np.array(avg_pre_rewards)
    post_rewards = np.array(avg_post_rewards)
    agreements = np.array(avg_agreements)
    entropies = np.array(avg_entropies)

    # Moving averages
    pre_smooth = moving_average(pre_rewards, window)
    post_smooth = moving_average(post_rewards, window)
    agree_smooth = moving_average(agreements, window)
    entropy_smooth = moving_average(entropies, window)

    plt.figure(figsize=(10, 5))

    plt.scatter(generations, pre_rewards, color="#1f77b4", s=10, alpha=0.4)
    plt.scatter(generations, post_rewards, color="#d62728", s=10, alpha=0.4)

    plt.plot(
        generations,
        pre_smooth,
        color="#1f77b4",
        linewidth=2,
        label="Pre-Learning Reward",
    )

    plt.plot(
        generations,
        post_smooth,
        color="#d62728",
        linewidth=2,
        label="Post-Learning Reward",
    )

    plt.xlabel("Generation")
    plt.ylabel("Reward")
    plt.title("MiniGrid Reward Trends")
    plt.xlim(0, generations.max())

    plt.legend(frameon=False)
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.savefig(
        "Data/Figures/Minigrid_Rewards.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

    plt.figure(figsize=(10, 5))

    plt.scatter(generations, agreements, color="#ff7f0e", s=10, alpha=0.4)
    plt.scatter(generations, entropies, color="#2ca02c", s=10, alpha=0.4)

    plt.plot(
        generations,
        agree_smooth,
        color="#ff7f0e",
        linewidth=2,
        label="Expert Agreement",
    )

    plt.plot(
        generations,
        entropy_smooth,
        color="#2ca02c",
        linewidth=2,
        label="Policy Entropy",
    )

    plt.xlabel("Generation")
    plt.ylabel("Value")
    plt.title("MiniGrid Behaviour Trends")
    plt.xlim(0, generations.max())

    plt.legend(frameon=False)
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.savefig(
        "Data/Figures/Minigrid_Behaviour.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()
    

def plot_assimilation_summary_panel(events, output_dir):

    if events.empty:
        print("No event data")
        return

    import matplotlib.pyplot as plt

    # ---- global styling (local to function) ----
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    total = len(events)
    assimilated = events["assimilation_generation"].notna().sum()
    not_assimilated = total - assimilated

    ax1.pie(
        [assimilated, not_assimilated],
        labels=["Assimilated", "Not Assimilated"],
        autopct="%1.1f%%",
        startangle=90,
        colors=["#2ecc71", "#e74c3c"],
        wedgeprops={"edgecolor": "white", "linewidth": 1}
    )

    ax1.set_title("Assimilation Rate")

    counts = events["persistent"].value_counts()

    persistent = counts.get(True, 0)
    non_persistent = counts.get(False, 0)

    ax2.pie(
        [persistent, non_persistent],
        labels=["Persistent", "Non-persistent"],
        autopct="%1.1f%%",
        startangle=90,
        colors=["#3498db", "#95a5a6"],
        wedgeprops={"edgecolor": "white", "linewidth": 1}
    )

    ax2.set_title("Persistence of Assimilation")

    fig.suptitle("MiniGrid Assimilation Summary", fontsize=16, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.92])

    plt.savefig(
        f"{output_dir}/MiniGrid_Assimilation_Summary_Panel.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def plot_lag_distribution(events, output_dir):
    lags = events["lag"].dropna()

    if len(lags) == 0:
        print("No assimilation lags found")
        return

    plt.figure(figsize=(10, 6))

    plt.hist(lags, bins=20)

    plt.title("Assimilation Lag Distribution")
    plt.xlabel("Lag (generations)")
    plt.ylabel("Count")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/MiniGrid_Assimilation_Lag_Distribution.png")
    plt.close()


def plot_learning_timeline(events, output_dir):
    df = events.dropna(subset=["assimilation_generation"])

    if df.empty:
        print("No assimilation events for timeline")
        return

    plt.figure(figsize=(10, 6))

    plt.scatter(
        df["learned_generation"],
        df["assimilation_generation"],
        alpha=0.4
    )

    max_gen = max(
        df["learned_generation"].max(),
        df["assimilation_generation"].max()
    )

    plt.plot(
        [0, max_gen],
        [0, max_gen],
        linestyle="--",
        label="Immediate assimilation"
    )

    plt.title("Learning vs Assimilation Timing")
    plt.xlabel("Learned Generation")
    plt.ylabel("Assimilation Generation")

    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{output_dir}/MiniGrid_Assimilation_Learning_Timeline.png")
    plt.close()
    
    
def plot_assimilation_stats():
    data_dir = "Data/Statistics/Assimilation"
    events_dir = f"{data_dir}/events.csv"
    modes_path = f"{data_dir}/modes.csv"
    output_dir = f"Data/Figures"
    
    events = pd.read_csv(events_dir)
    modes = pd.read_csv(modes_path)

    plot_assimilation_summary_panel(events, output_dir)
    plot_lag_distribution(events, output_dir)
    plot_learning_timeline(events, output_dir)


def main() -> None:
    data = pd.read_csv("Data/Statistics/minigrid_stats.csv")
    grouped = data.groupby("generation").mean(numeric_only=True)
    post_rewards = grouped["post_learn_reward"]
    pre_rewards = grouped["pre_learn_reward"]
    agreements = grouped["agreement"]
    entropies = grouped["entropy"]

    plot_metrics(post_rewards, pre_rewards, agreements, entropies)


if __name__ == "__main__":
    main()