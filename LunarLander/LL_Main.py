import pygad
import numpy as np
import csv
from LL_Constants import gene_space
from LL_Fitness import fitness_function

POPULATION_SIZE = 20
NUM_GENERATIONS = 60
NUM_PARENTS = 4

def start_file_tracking():
    csv_file_avg = open("ll_baldwin_avgs.csv", "w", newline="")
    avg_writer = csv.writer(csv_file_avg)
    avg_writer.writerow(
        [
            "generation",
            "avg_untrained_reward",
            "avg_trained_reward",
            "avg_learning_delta",
        ]
    )
    csv_file_gen = open("ll_baldwin_stats.csv", "w", newline="")
    gen_writer = csv.writer(csv_file_gen)
    gen_writer.writerow(
        [
            "generation",
            "individual",
            "untrained_reward",
            "trained_reward",
            "learning_delta",
            "action_prob_0",
            "action_prob_1",
            "action_prob_2",
            "action_prob_3"
        ]
    )
    return csv_file_avg, csv_file_gen, avg_writer, gen_writer

def on_generation(ga_instance):
    last_gen = ga_instance.generations_completed - 1
    
    gen_metrics = [m for m in ga_instance.metrics if m['generation'] == last_gen]
    
    if gen_metrics:
        avg_untrained = np.mean([m['untrained'] for m in gen_metrics])
        avg_trained = np.mean([m['trained'] for m in gen_metrics])
        avg_delta = np.mean([m['learning_delta'] for m in gen_metrics])
        
        print("\n" + "="*30)
        print(f"SUMMARY FOR GENERATION {last_gen}")
        print(f"Avg Untrained: {avg_untrained:.1f}")
        print(f"Avg Trained:   {avg_trained:.1f}")
        print(f"Avg Delta:     {avg_delta:.1f}")
        print("="*30 + "\n")

def main():
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

    csv_file_avg, csv_file_gen, avg_writer, gen_writer = start_file_tracking()
    ga_instance.run()

    for gen in range(NUM_GENERATIONS):
        gen_metrics = [m for m in ga_instance.metrics if m['generation'] == gen]
        if gen_metrics:
            for metric in gen_metrics:
                all_actions = [a for ep in metric['trained_actions'] for a in ep]
                action_counts = np.bincount(all_actions, minlength=4)
                action_probs = action_counts / action_counts.sum()

                gen_writer.writerow(
                    [
                        gen,
                        metric['solution_idx'],
                        metric['untrained'],
                        metric['trained'],
                        metric['learning_delta'],
                        *action_probs.tolist()
                    ]
                )
            avg_untrained = np.mean([m['untrained'] for m in gen_metrics])
            avg_trained = np.mean([m['trained'] for m in gen_metrics])
            avg_delta = np.mean([m['learning_delta'] for m in gen_metrics])
            avg_writer.writerow(
                [
                    gen,
                    avg_untrained,
                    avg_trained,
                    avg_delta,
                ]
            )
    csv_file_gen.close()
    csv_file_avg.close()
    solution, fitness, _ = ga_instance.best_solution()
    print(f"\nBest Solution Fitness: {fitness:.2f}")

if __name__ == "__main__":
    main()
