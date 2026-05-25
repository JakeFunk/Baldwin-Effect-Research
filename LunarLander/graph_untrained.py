import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("60_40_split_STATS.csv")

# Average untrained_reward per generation
gen_avg = df.groupby("generation")["untrained_reward"].mean().sort_index()

# Compute increase between generations
gen_increase = gen_avg.diff()

# Drop first NaN
gen_increase = gen_increase.dropna()


cumulative = gen_increase.cumsum()

plt.figure()
plt.plot(cumulative.index, cumulative.values)
plt.xlabel("Generation")
plt.ylabel("Cumulative Increase")
plt.title("Cumulative Untrained Reward Improvement")
plt.show()

