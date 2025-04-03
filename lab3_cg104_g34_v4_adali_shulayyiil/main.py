from matplotlib import pyplot as plt
from genetic_algorithm import GeneticAlgorithm
from functions import booth_2d
if __name__ == "__main__":
    # TODO Experiment 1...
    ga = GeneticAlgorithm(
        population_size=5000,
        mutation_rate=0.1,
        mutation_strength=0.1,
        crossover_rate=0.7,
        num_generations=25,
        fitness_function=booth_2d
    )
    best_solutions, best_fitness_values, average_fitness_values = ga.evolve(seed=420)
    print("Best Solutions:", best_solutions)
    print("Best Fitness Values:", best_fitness_values)
    print("Average Fitness Values:", average_fitness_values)
    print("\n ************************************** \n")
    print("\nBest Solution:",best_solutions[best_fitness_values.index(min(best_fitness_values))])
    print("\nBest Fitness:",min(best_fitness_values))
    
     # Plotting the results
    generations = list(range(len(best_fitness_values)))
    
    plt.figure(figsize=(10, 6))
    
    # Plot best fitness values with log scale for y-axis
    plt.subplot(2, 1, 1)
    plt.semilogy(generations, best_fitness_values, 'b-', linewidth=2, marker='o', label='Best Fitness')
    plt.title('Best Fitness Value per Generation (Log Scale)')
    plt.xlabel('Generation')
    plt.ylabel('Fitness Value (log scale)')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    
    # Plot average fitness values with log scale for y-axis
    plt.subplot(2, 1, 2)
    plt.semilogy(generations, average_fitness_values, 'r-', linewidth=2, marker='x', label='Average Fitness')
    plt.semilogy(generations, best_fitness_values, 'b-', linewidth=2, marker='o', label='Best Fitness')
    plt.title('Fitness Values per Generation (Log Scale)')
    plt.xlabel('Generation')
    plt.ylabel('Fitness Value (log scale)')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('genetic_algorithm_results.png')
    plt.show()
    