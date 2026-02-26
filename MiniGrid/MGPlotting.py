import matplotlib.pyplot as plt

def plot(avg_pre_rewards, avg_post_rewards, avg_gains, avg_agreements, avg_entropies):
    generations = range(len(avg_pre_rewards))

    # Pre-learning reward
    plt.figure(figsize=(10, 4))
    plt.plot(generations, avg_pre_rewards, marker="o")
    plt.xlabel("Generation")
    plt.ylabel("Avg Pre-Learn Reward")
    plt.title("Average Pre-Learning Reward")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Post-learning reward
    plt.figure(figsize=(10, 4))
    plt.plot(generations, avg_post_rewards, marker="o")
    plt.xlabel("Generation")
    plt.ylabel("Avg Post-Learn Reward")
    plt.title("Average Post-Learning Reward")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Learning gain
    plt.figure(figsize=(10, 4))
    plt.plot(generations, avg_gains, marker="o")
    plt.xlabel("Generation")
    plt.ylabel("Avg Learning Gain")
    plt.title("Average Learning Gain")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Agreement
    plt.figure(figsize=(10, 4))
    plt.plot(generations, avg_agreements, marker="o")
    plt.xlabel("Generation")
    plt.ylabel("Avg Agreement")
    plt.title("Average Agreement with Expert")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Entropy
    plt.figure(figsize=(10, 4))
    plt.plot(generations, avg_entropies, marker="o")
    plt.xlabel("Generation")
    plt.ylabel("Avg Policy Entropy")
    plt.title("Average Pre-Learning Policy Entropy")
    plt.grid(True)
    plt.tight_layout()
    plt.show()