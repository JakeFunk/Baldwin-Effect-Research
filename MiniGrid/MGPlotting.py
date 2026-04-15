from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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


def plot_metrics(avg_pre_rewards, avg_agreements, avg_entropies, window=10):
    generations = np.array(avg_pre_rewards.index)

    pre_rewards = np.array(avg_pre_rewards)
    agreements = np.array(avg_agreements)
    entropies = np.array(avg_entropies)

    # Correct moving averages
    pre_smooth = moving_average(pre_rewards, window)
    agree_smooth = moving_average(agreements, window)
    entropy_smooth = moving_average(entropies, window)

    plt.figure(figsize=(10, 5))

    # Raw points
    plt.scatter(generations, pre_rewards, color="#1f77b4", s=10, alpha=0.4)
    plt.scatter(generations, agreements, color="#ff7f0e", s=10, alpha=0.4)
    plt.scatter(generations, entropies, color="#2ca02c", s=10, alpha=0.4)

    # Smoothed lines
    plt.plot(
        generations,
        pre_smooth,
        color="#1f77b4",
        linewidth=2,
        label="Pre-Learning Reward",
    )
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
    plt.title("MiniGrid Baldwin Effect Detection")
    plt.xlim(0, generations.max())

    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)

    plt.grid(True, linestyle="--", alpha=0.5)
    plt.subplots_adjust(bottom=0.25)

    plt.savefig("Data/Figures/Minigrid_Metrics.png", dpi=300, bbox_inches="tight")
    plt.show()
    

def plot_policy_analysis(csv_path):
    df = pd.read_csv(csv_path)

    summary = (
        df.groupby(["target_gen", "target_phase"])["agreement"]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(14, 7))
    sns.set(style="whitegrid", context="talk")

    before = summary[summary["target_phase"] == "before"]
    after = summary[summary["target_phase"] == "after"]

    plt.plot(
        before["target_gen"],
        before["agreement"],
        marker="o",
        linewidth=2,
        label="Starting Policy"
    )

    plt.plot(
        after["target_gen"],
        after["agreement"],
        marker="s",
        linewidth=2,
        label="Learned Policy"
    )

    plt.title("Policy Agreement Throughout Evolution")
    plt.xlabel("Generation")
    plt.ylabel("Agreement With Final Generation")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main() -> None:
    data = pd.read_csv("Data/Statistics/minigrid_stats.csv")
    grouped = data.groupby("generation").mean(numeric_only=True)
    pre_rewards = grouped["pre_learn_reward"]
    agreements = grouped["agreement"]
    entropies = grouped["entropy"]

    plot_metrics(pre_rewards, agreements, entropies)


if __name__ == "__main__":
    main()
