import pygad
import numpy as np
import csv
import matplotlib.pyplot as plt
from LL_Constants import gene_space
from LL_Fitness import fitness_function
import torch

POPULATION_SIZE = 20
NUM_GENERATIONS = 60
NUM_PARENTS = 4



def start_file_tracking():
    csv_file_avg = open("TRAINED_PREF_split_AVG.csv", "w", newline="")
    avg_writer = csv.writer(csv_file_avg)
    avg_writer.writerow([
        "generation", "avg_untrained_reward", "avg_trained_reward", "avg_learning_delta",
        "avg_untrained_agreement_expert", "avg_untrained_agreement_agent", 
        "avg_trained_agreement_expert", "avg_trained_agreement_agent",
        "avg_num_layers", "most_common_activation", "activation_diversity"
    ])

    csv_file_gen = open("TRAINED_PREF_split_STATS.csv", "w", newline="")
    gen_writer = csv.writer(csv_file_gen)
    gen_writer.writerow([
        "generation", "individual", "activations", "num_layers",
        "untrained_reward", "trained_reward", "learning_delta",
        "untrained_agreement_expert", "untrained_agreement_agent",
        "trained_agreement_expert", "trained_agreement_agent",
        "action_prob_0", "action_prob_1", "action_prob_2", "action_prob_3"
    ])
    return csv_file_avg, csv_file_gen, avg_writer, gen_writer

def plot(avg_pre_rewards, avg_post_rewards, avg_gains, u_agr_age, t_agr_age, avg_entropies, avg_layers):
    generations = range(len(avg_pre_rewards))
    
    # Pre-learning reward
    plt.figure(figsize=(10, 4))
    plt.plot(generations, avg_pre_rewards, marker="o")
    plt.xlabel("Generation")
    plt.ylabel("Avg Pre-Learn Reward")
    plt.title("Average Pre-Learning Reward")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plot_pre_learning_reward_TRAINED_PREF.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Post-learning reward
    plt.figure(figsize=(10, 4))
    plt.plot(generations, avg_post_rewards, marker="o")
    plt.xlabel("Generation")
    plt.ylabel("Avg Post-Learn Reward")
    plt.title("Average Post-Learning Reward")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plot_post_learning_reward_TRAINED_PREF.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Learning gain
    plt.figure(figsize=(10, 4))
    plt.plot(generations, avg_gains, marker="o")
    plt.xlabel("Generation")
    plt.ylabel("Avg Learning Gain")
    plt.title("Average Learning Gain")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plot_learning_gain_TRAINED_PREF.png", dpi=300, bbox_inches='tight')
    plt.close()
   
    #Agreement
    plt.figure(figsize=(10, 4))
    plt.plot(generations, u_agr_age, label="Innate (Untrained)", marker="o")
    plt.plot(generations, t_agr_age, label="Learned (Trained)", marker="s")
    plt.xlabel("Generation")
    plt.ylabel("Avg Agreement (Agent Steps)")
    plt.title("Active Policy Agreement with Expert")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plot_agreement_ACTIVE_TRAINED_PREF.png", dpi=300)
    plt.close()


    # Entropy
    plt.figure(figsize=(10, 4))
    plt.plot(generations, avg_entropies, marker="o")
    plt.xlabel("Generation")
    plt.ylabel("Avg Policy Entropy")
    plt.title("Average Pre-Learning Policy Entropy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plot_entropy_DELTA.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Average number of layers
    plt.figure(figsize=(10, 4))
    plt.plot(generations, avg_layers, marker="o")
    plt.xlabel("Generation")
    plt.ylabel("Avg Number of Layers")
    plt.title("Average Number of Hidden Layers")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plot_avg_layers_TRAINED_PREF.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n✓ All plots saved successfully!")

def on_generation(ga_instance):
    last_gen = ga_instance.generations_completed - 1
    gen_metrics = [m for m in ga_instance.metrics if m['generation'] == last_gen]

    if gen_metrics:
        avg_untrained = np.mean([m['untrained'] for m in gen_metrics])
        avg_trained = np.mean([m['trained'] for m in gen_metrics])
        avg_delta = np.mean([m['learning_delta'] for m in gen_metrics])

        avg_u_agr_exp = np.mean([m['untrained_agreement_expert'] for m in gen_metrics])
        avg_u_agr_age = np.mean([m['untrained_agreement_agent'] for m in gen_metrics])
        avg_t_agr_exp = np.mean([m['trained_agreement_expert'] for m in gen_metrics])
        avg_t_agr_age = np.mean([m['trained_agreement_agent'] for m in gen_metrics])
        
        avg_num_layers = np.mean([m['num_layers'] for m in gen_metrics])
        all_activations = [m['activations'] for m in gen_metrics]
        activation_counts = {}

        for act in all_activations:
            activation_counts[act] = activation_counts.get(act, 0) + 1
        
        most_common_activation = max(activation_counts, key=activation_counts.get)
        activation_diversity = len(activation_counts) / len(gen_metrics) 

        for metric in gen_metrics:
            all_actions = [a for ep in metric['untrained_actions'] for a in ep]
            if len(all_actions) > 0:
                action_counts = np.bincount(all_actions, minlength=4)
                action_probs = action_counts / action_counts.sum()
            else:
                action_probs = np.zeros(4)

            ga_instance.gen_writer.writerow([
                last_gen,
                metric['solution_idx'],
                metric['activations'],
                metric['num_layers'],
                metric['untrained'],
                metric['trained'],
                metric['learning_delta'],
                metric['untrained_agreement_expert'], 
                metric['untrained_agreement_agent'],
                metric['trained_agreement_expert'], 
                metric['trained_agreement_agent'],
                *action_probs.tolist()
            ])
                
        ga_instance.avg_writer.writerow([
            last_gen,
            avg_untrained,
            avg_trained,
            avg_delta,
            avg_u_agr_exp, 
            avg_u_agr_age,
            avg_t_agr_exp,
            avg_t_agr_age,
            avg_num_layers,
            most_common_activation,
            activation_diversity
        ])

        ga_instance.csv_file_avg.flush()
        ga_instance.csv_file_gen.flush()

        print("\n" + "="*30)
        print(f"SUMMARY FOR GENERATION {last_gen}")
        print(f"Avg Untrained: {avg_untrained:.1f} | Agr(U-Agent): {avg_u_agr_age:.2%}")
        print(f"Avg Trained:   {avg_trained:.1f} | Agr(T-Agent): {avg_t_agr_age:.2%}")
        print(f"Avg Delta:     {avg_delta:.1f}")
        print(f"Avg Layers:    {avg_num_layers:.1f}")
        print(f"Most Common:   {most_common_activation} | Diversity: {activation_diversity:.2f}")
        print("="*30 + "\n")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n" + "="*30)
    print(f"TRAINING DEVICE: {device.type.upper()}")
    if device.type == 'cuda':
        print(f"GPU NAME: {torch.cuda.get_device_name(0)}")
    print("="*30 + "\n")
    csv_file_avg, csv_file_gen, avg_writer, gen_writer = start_file_tracking()
    ga_instance = pygad.GA(
        num_generations=NUM_GENERATIONS,
        num_parents_mating=NUM_PARENTS,
        fitness_func=fitness_function,
        sol_per_pop=POPULATION_SIZE,
        num_genes=len(gene_space),
        gene_space=gene_space,
        parent_selection_type="tournament",
        keep_parents=2,
        crossover_type="single_point",
        mutation_type="random",
        mutation_percent_genes=5,
        on_generation=on_generation,
        allow_duplicate_genes=True,
    )
    
    ga_instance.metrics = []
    ga_instance.avg_writer = avg_writer
    ga_instance.gen_writer = gen_writer
    ga_instance.csv_file_avg = csv_file_avg
    ga_instance.csv_file_gen = csv_file_gen
    
    try:
        ga_instance.run()
    finally:
        csv_file_gen.close()
        csv_file_avg.close()
    
    solution, fitness, _ = ga_instance.best_solution()
    print(f"\nBest Solution Fitness: {fitness:.2f}")
    
    avg_pre_rewards = []
    avg_post_rewards = []
    avg_gains = []
    avg_u_agreements_age = []
    avg_t_agreements_age = []
    avg_entropies = []
    avg_layers = []
    
    with open("TRAINED_PREF_split_AVG.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            avg_pre_rewards.append(float(row['avg_untrained_reward']))
            avg_post_rewards.append(float(row['avg_trained_reward']))
            avg_gains.append(float(row['avg_learning_delta']))
            
            avg_u_agreements_age.append(float(row['avg_untrained_agreement_agent']))
            avg_t_agreements_age.append(float(row['avg_trained_agreement_agent']))

            avg_layers.append(float(row['avg_num_layers']))
    
    for gen in range(NUM_GENERATIONS):
        gen_entropies = []
        with open("TRAINED_PREF_split_STATS.csv", "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if int(row['generation']) == gen:
                    probs = [
                        float(row['action_prob_0']),
                        float(row['action_prob_1']),
                        float(row['action_prob_2']),
                        float(row['action_prob_3'])
                    ]
                    entropy = -np.sum([p * np.log(p + 1e-10) for p in probs if p > 0])
                    gen_entropies.append(entropy)
        
        if gen_entropies:
            avg_entropies.append(np.mean(gen_entropies))
        else:
            avg_entropies.append(0)
    
    plot(avg_pre_rewards, avg_post_rewards, avg_gains, avg_u_agreements_age, avg_t_agreements_age, avg_entropies, avg_layers)


if __name__ == "__main__":
    main()

