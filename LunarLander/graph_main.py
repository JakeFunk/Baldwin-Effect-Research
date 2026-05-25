from re import A
from matplotlib import figure
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def calculate_entropy(df):
    probs = df[['action_prob_0', 'action_prob_1', 'action_prob_2', 'action_prob_3']]
    entropy = - (probs * np.log2(probs + 1e-9)).sum(axis=1)
    return entropy

def smooth_line(generations, y, degree=3):
    coeffs = np.polyfit(generations, y, degree)
    poly = np.poly1d(coeffs)
    x_smooth = np.linspace(generations.min(), generations.max(), 300)
    y_smooth = poly(x_smooth)
    return x_smooth, y_smooth

#def normalize_ll(series):
#        return 2 * (series - series.min()) / (series.max() - series.min()) - 1

def normalize_ll(series):
    return (series - series.min()) / (series.max() - series.min())


def plot_lunarlander(untrained_rewards, untrained_agreement):
    generations = np.array(untrained_rewards.index)

    fig, ax1 = plt.subplots(figsize=(10,5))

    # Reward axis
    r1 = ax1.scatter(
        generations,
        untrained_rewards,
        color="#069AF3",
        s=10,
        alpha=0.55,
        label="Untrained Reward"
    )
    x_s, y_s = smooth_line(generations, untrained_rewards)
    l1, = ax1.plot(x_s, y_s, color="#069AF3", linewidth=2, label="Reward Trend")

    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Reward")

    # Agreement axis
    ax2 = ax1.twinx()
    r2 = ax2.scatter(
        generations,
        untrained_agreement,
        color="#F35F06",
        s=10,
        alpha=0.55,
        label="Untrained Agreement (%)"
    )
    x_s, y_s = smooth_line(generations, untrained_agreement)
    l2, = ax2.plot(x_s, y_s, color="#F35F06", linewidth=2, label="Agreement Trend")

    ax2.set_ylabel("Agreement (%)")

    # ---- combine legends from both axes ----
    handles = [r1, l1, r2, l2]
    labels = [h.get_label() for h in handles]

    ax1.legend(handles, labels, loc='upper left')

    plt.title("Lunar Lander Baldwin Effect Detection")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig("LunarLander_DualAxis.png", dpi=300, bbox_inches='tight')
    plt.show()




"""
def plot_lunarlander(untrained_rewards, untrained_agreement):
    generations = np.array(untrained_rewards.index)

    fig,ax1 = plt.subplots(figsize=(10,5))

    # Reward axis
    ax1.scatter(generations, untrained_rewards, color="#069AF3", s=10, alpha=0.55)
    x_s, y_s = smooth_line(generations, untrained_rewards)
    ax1.plot(x_s, y_s, color="#069AF3", linewidth=2)
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Reward")

    # Agreement axis
    ax2 = ax1.twinx()
    ax2.scatter(generations, untrained_agreement, color="#F35F06", s=10, alpha=0.55)
    x_s, y_s = smooth_line(generations, untrained_agreement)
    ax2.plot(x_s, y_s, color="#F35F06", linewidth=2)
    ax2.set_ylabel("Agreement (%)")

    ax1.legend(loc='upper left')
    ax2.legend(loc='upper left')

    plt.title("Lunar Lander Baldwin Effect Detection")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig("LunarLander_DualAxis.png", dpi=300, bbox_inches='tight')
    plt.show()
"""
"""
def plot_lunarlander(untrained_rewards, untrained_agreement):    
    generations = np.array(untrained_rewards.index)
    
    untrained_rewards_norm = untrained_rewards #normalize_ll(untrained_rewards)
    untrained_agreement_norm = untrained_agreement #normalize_ll(untrained_agreement)

    plt.figure(figsize=(10,5))
    plt.scatter(generations, untrained_rewards_norm, color="#069AF3", label="Untrained Reward", s=10,alpha=0.55)
    plt.scatter(generations, untrained_agreement_norm, color="#F35F06", label="Untrained Agreement", s=10,alpha=0.55)

    x_s, y_s = smooth_line(generations, untrained_rewards_norm)
    plt.plot(x_s, y_s, color="#069AF3", linewidth=2)
    x_s, y_s = smooth_line(generations, untrained_agreement_norm)
    plt.plot(x_s, y_s, color="#F35F06", linewidth=2)

    plt.xlabel("Generation")
    plt.ylabel("Value")
    plt.title("Lunar Lander Baldwin Effect Detection")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig("LunarLander_Normalized.png", dpi=300, bbox_inches='tight')
    plt.show()
"""

def plot_entropy(entropy_series):
    generations = np.array(entropy_series.index)
    plt.figure(figsize=(10,5))
    plt.scatter(generations, entropy_series, color="#15B01A", s=10, alpha=0.55)
    xs, ys = smooth_line(generations, entropy_series.values, degree=2)
    plt.plot(xs, ys, color="#15B01A", linewidth=2, label="Policy Entropy")
    plt.xlabel("Generation")
    plt.ylabel("Entropy")
    plt.title("Action Entropy Over Generations")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("LunarLander_Entropy.png",dpi=300,bbox_inches='tight')
    plt.show()




def plot_trained_performance(trained_rewards_series):
    """
    Plots the Mean Trained Reward over generations.
    Expects a Series where the index is 'generation'.
    """
    generations = np.array(trained_rewards_series.index)
    
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Scatter for the average points
    r1 = ax1.scatter(
        generations, 
        trained_rewards_series, 
        color="#9400D3",  # Dark Violet
        s=15, 
        alpha=0.6, 
        label="Mean Trained Reward"
    )

    # Trendline
    x_s, y_s = smooth_line(generations, trained_rewards_series, degree=3)
    l1, = ax1.plot(x_s, y_s, color="#9400D3", linewidth=2, label="Learning Trend")

    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Trained Reward (Fitness)")
    ax1.set_title("Average Training Performance per Generation")
    
    # Grid and Legend
    ax1.grid(True, linestyle="--", alpha=0.4)
    handles = [r1, l1]
    labels = [h.get_label() for h in handles]
    ax1.legend(handles, labels, loc='upper left')

    plt.savefig("Trained_Reward_Trend.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_reward_comparison(untrained_rewards, trained_rewards):
    """
    Plots Pre-training (Untrained) vs Post-training (Trained) rewards 
    on the same axis to visualize the learning gap.
    """
    generations = np.array(untrained_rewards.index)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(generations, untrained_rewards, color="#069AF3", s=15, alpha=0.4, label="Pre-Training Reward")
    x_u, y_u = smooth_line(generations, untrained_rewards, degree=3)
    ax.plot(x_u, y_u, color="#069AF3", linewidth=2.5)

    ax.scatter(generations, trained_rewards, color="#9400D3", s=15, alpha=0.4, label="Post-Training Reward")
    x_t, y_t = smooth_line(generations, trained_rewards, degree=3)
    ax.plot(x_t, y_t, color="#9400D3", linewidth=2.5)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Reward")
    ax.set_title("Reward Improvement: Pre- vs. Post-Training")
    
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc='upper left')

    plt.savefig("Reward_Comparison.png", dpi=300, bbox_inches='tight')
    plt.show()


def main():
    ll_data = pd.read_csv('60_40_split_STATS.csv')

    gen_stats = ll_data.groupby('generation').mean(numeric_only=True)

    untrained_rewards = gen_stats['untrained_reward']
    untrained_agreement = gen_stats['untrained_agreement'] * 100
    
    trained_rewards = gen_stats['trained_reward'] 
    
    plot_reward_comparison(untrained_rewards, trained_rewards)

if __name__ == "__main__":
    main()

