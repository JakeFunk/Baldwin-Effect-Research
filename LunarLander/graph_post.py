import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np

def smooth_line(generations, y, degree=3):
    coeffs = np.polyfit(generations, y, degree)
    poly = np.poly1d(coeffs)
    x_smooth = np.linspace(generations.min(), generations.max(), 300)
    y_smooth = poly(x_smooth)
    return x_smooth, y_smooth

def plot_manual_training_reward(reward_list):
    """
    Plots training reward from a list of values.
    Assumes index corresponds to generation number.
    """
    rewards = pd.Series(reward_list)
    generations = np.array(rewards.index)

    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Scatter points
    r1 = ax1.scatter(
        generations,
        rewards,
        color="#D35400",
        s=15,
        alpha=0.6,
        label="Avg Trained Reward"
    )

    # Trendline
    x_s, y_s = smooth_line(generations, rewards, degree=3)
    l1, = ax1.plot(x_s, y_s, color="#D35400", linewidth=2.5, label="Learning Trend")

    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Average Reward")
    ax1.set_title("Training Reward Increase Over 60 Generations")
    
    ax1.grid(True, linestyle="--", alpha=0.4)
    handles = [r1, l1]
    labels = [h.get_label() for h in handles]
    ax1.legend(handles, labels, loc='upper left')

    plt.tight_layout()
    plt.savefig("Training_Increase_Manual.png", dpi=300)
    plt.show()

def main():
    trained_rewards = [
        42.74557492, 85.45721368, 86.48056345, 66.67092835, 18.32104483,
        65.48523114, 60.91176358, 100.2722488, 56.31292492, 45.78433385,
        76.76896664, 70.75078529, 99.85500804, 101.6368135, 133.9649261,
        106.0203136, 97.75999831, 115.5244474, 105.5330977, 77.1198577,
        85.22382338, 132.5749014, 110.5074532, 123.7442401, 109.8446218,
        136.6478916, 134.3486074, 97.01296513, 119.9058241, 105.9496543,
        115.4499069, 97.72707027, 83.07586761, 134.0178223, 110.6598543,
        102.5391455, 100.4382321, 118.3853626, 101.9765382, 113.5642614,
        102.9154327, 118.7835262, 155.8697414, 129.1245446, 127.5386903,
        125.2935802, 93.70700552, 97.22049543, 100.7220969, 102.4719073,
        116.0328689, 118.6753644, 95.63165283, 137.9883639, 141.6351164,
        163.1338815, 131.9060487, 112.1992098, 134.5455479, 110.8638294
    ]
    plot_manual_training_reward(trained_rewards)

if __name__ == "__main__":
    main()
