import pygad
import numpy as np
from LL_Constants import gene_space
from LL_Fitness import fitness_function

POPULATION_SIZE = 20
NUM_GENERATIONS = 30
NUM_PARENTS = 6

def on_generation(ga):
    print(f"\n=== Generation {ga.generations_completed} ===")
    print("Best Fitness:", ga.best_solution()[1])

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
        allow_duplicate_genes=True
    )

    ga_instance.run()

    # Best solution
    solution, fitness, _ = ga_instance.best_solution()

    print("\n========= FINAL RESULT =========")
    print("Best Fitness:", fitness)

    # Save best genome
    np.save("best_genome.npy", solution)

    # Save metrics log
    if hasattr(ga_instance, "metrics"):
        np.save("metrics.npy", ga_instance.metrics)

    print("Saved best genome & metrics.")

if __name__ == "__main__":
    main()
