from collections import defaultdict, Counter
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
    

def plot_event_locations(events, flat_env, side, out_path, title="Assimilated state locations"):
    import numpy as np
    import matplotlib.pyplot as plt

    grid = np.array(flat_env).reshape(side, side)
    counts = np.zeros((side, side))

    for e in events:
        x, y, _d = e["state"]
        counts[y, x] += 1

    fig, ax = plt.subplots(figsize=(6, 6))

    # background: show walls/lava/goal
    bg = np.zeros((side, side, 3))
    bg[grid == 1] = [0.3, 0.3, 0.3]   # wall - gray
    bg[grid == 2] = [1.0, 0.4, 0.4]   # lava - red
    bg[grid == 5] = [0.4, 1.0, 0.4]   # goal - green
    bg[grid == 0] = [1.0, 1.0, 1.0]   # empty - white

    ax.imshow(bg, origin="upper")

    # overlay counts as text/heatmap on top
    masked = np.ma.masked_where(counts == 0, counts)
    ax.imshow(masked, cmap="Blues", alpha=0.6, origin="upper")

    for y in range(side):
        for x in range(side):
            if counts[y, x] > 0:
                ax.text(x, y, int(counts[y, x]), ha="center", va="center",
                        fontsize=10, color="black")

    ax.set_title(title)
    ax.set_xticks(range(side))
    ax.set_yticks(range(side))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved event location plot to {out_path}")
    

def plot_persistence_distribution(events, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)

    for ax, action in zip(axes, (0, 1, 2)):
        vals = [e["persistence"] for e in events if e["learned_action"] == action]
        ax.hist(vals, bins=20, color="steelblue", edgecolor="black")
        ax.set_title(f"Action {action}")
        ax.set_xlabel("Persistence (generations)")

    axes[0].set_ylabel("Count")
    fig.suptitle("Distribution of divergence persistence by action")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_origin_gen_timeline(events, out_path, threshold=5):
    fig, ax = plt.subplots(figsize=(10, 5))

    colors = {0: "crimson", 1: "steelblue", 2: "seagreen"}
    labels = {0: "turn left", 1: "turn right", 2: "forward"}

    for action in (0, 1, 2):
        xs = [e["origin_gen"] for e in events if e["learned_action"] == action]
        ys = [e["persistence"] for e in events if e["learned_action"] == action]
        ax.scatter(xs, ys, label=labels[action], color=colors[action], alpha=0.6, s=30)

    ax.axvline(threshold, color="gray", linestyle="--", label=f"early/late split (gen {threshold})")
    ax.set_xlabel("Origin generation")
    ax.set_ylabel("Persistence (generations)")
    ax.set_title("Divergence persistence over time, by action")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


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