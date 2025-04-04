from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from genetic_algorithm import GeneticAlgorithm
from functions import booth_2d
import os

def run_single_experiment(params, seed=None):
    """Run a single experiment with the given parameters and seed."""
    ga = GeneticAlgorithm(
        population_size=params['population_size'],
        mutation_rate=params['mutation_rate'],
        mutation_strength=params['mutation_strength'],
        crossover_rate=params['crossover_rate'],
        num_generations=params['num_generations'],
        fitness_function=booth_2d
    )
    
    best_solutions, best_fitness_values, average_fitness_values = ga.evolve(seed=seed)
    
    final_best_solution = best_solutions[best_fitness_values.index(min(best_fitness_values))]
    final_best_fitness = min(best_fitness_values)
    
    return {
        'best_solution': final_best_solution,
        'best_fitness': final_best_fitness,
        'best_fitness_history': best_fitness_values,
        'avg_fitness_history': average_fitness_values
    }

def plot_fitness_history(results_dict, title, filename):
    """Plot fitness values across generations for multiple experiment results."""
    plt.figure(figsize=(12, 8))
    
    # Plot best fitness values
    plt.subplot(2, 1, 1)
    for label, result in results_dict.items():
        generations = list(range(len(result['best_fitness_history'])))
        plt.semilogy(generations, result['best_fitness_history'], linewidth=2, marker='o', markersize=3, label=f"{label}")
    
    plt.title(f'Best Fitness Value per Generation - {title}')
    plt.xlabel('Generation')
    plt.ylabel('Fitness Value (log scale)')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    
    # Plot average fitness values
    plt.subplot(2, 1, 2)
    for label, result in results_dict.items():
        generations = list(range(len(result['avg_fitness_history'])))
        plt.semilogy(generations, result['avg_fitness_history'], linewidth=2, marker='x', markersize=3, label=f"{label}")
    
    plt.title(f'Average Fitness Value per Generation - {title}')
    plt.xlabel('Generation')
    plt.ylabel('Fitness Value (log scale)')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'results/{filename}.png')
    plt.close()

def experiment_1_parameter_tuning():
    """
    1. Finding genetic algorithm parameters:
    Run the algorithm with different parameters until you find a set of parameters 
    that obtains a good fitness value.
    """
    print("\n======= EXPERIMENT 1: PARAMETER TUNING =======")
    
    # Define parameter combinations to test
    parameter_combinations = [
        {'population_size': 1000, 'mutation_rate': 0.1, 'mutation_strength': 0.1, 'crossover_rate': 0.7, 'num_generations': 35},
        {'population_size': 2000, 'mutation_rate': 0.1, 'mutation_strength': 0.1, 'crossover_rate': 0.7, 'num_generations': 20},
        {'population_size': 1000, 'mutation_rate': 0.2, 'mutation_strength': 0.1, 'crossover_rate': 0.7, 'num_generations': 20},
        {'population_size': 1000, 'mutation_rate': 0.1, 'mutation_strength': 0.2, 'crossover_rate': 0.7, 'num_generations': 30},
        {'population_size': 1000, 'mutation_rate': 0.1, 'mutation_strength': 0.1, 'crossover_rate': 0.8, 'num_generations': 20},
        {'population_size': 5000, 'mutation_rate': 0.1, 'mutation_strength': 0.1, 'crossover_rate': 0.7, 'num_generations': 10},
        {'population_size': 5000, 'mutation_rate': 0.1, 'mutation_strength': 0.1, 'crossover_rate': 0.7, 'num_generations': 30},
    ]
    
    results = []
    
    for params in parameter_combinations:
        print(f"Testing parameters: {params}")
        result = run_single_experiment(params, seed=42)
        
        param_result = {
            'population_size': params['population_size'],
            'mutation_rate': params['mutation_rate'],
            'mutation_strength': params['mutation_strength'],
            'crossover_rate': params['crossover_rate'],
            'best_solution': str(result['best_solution']),
            'best_fitness': result['best_fitness']
        }
        results.append(param_result)
    
    # Create results DataFrame and save to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv('results/parameter_tuning_results.csv', index=False)
    
    print("\nParameter Tuning Results:")
    print(results_df.to_string())
    
    # Find best parameter combination
    best_idx = results_df['best_fitness'].idxmin()
    best_params = {key: results_df.iloc[best_idx][key] for key in ['population_size', 'mutation_rate', 'mutation_strength', 'crossover_rate']}
    best_params['num_generations'] = 30  # Keep num_generations constant
    
    print(f"\nBest parameter combination: {best_params}")
    print(f"Best fitness value: {results_df.iloc[best_idx]['best_fitness']}")
    
    return best_params

def experiment_2_randomness(best_params):
    """
    2. Randomness in genetic algorithm:
    Run the algorithm with the best parameters using 5 different random seeds.
    Then rerun with decreasing population sizes.
    """
    print("\n======= EXPERIMENT 2: RANDOMNESS ANALYSIS =======")
    
    seeds = [69, 123, 420, 789, 1010]
    population_sizes = [
        best_params['population_size'],
        int(best_params['population_size'] * 0.5),  # 50%
        int(best_params['population_size'] * 0.25), # 25%
        int(best_params['population_size'] * 0.1)   # 10%
    ]
    
    # First part: Run with best parameters using different seeds
    seed_results = []
    
    for seed in seeds:
        params = best_params.copy()
        result = run_single_experiment(params, seed=seed)
        seed_results.append({
            'seed': seed,
            'best_solution': result['best_solution'],
            'best_fitness': result['best_fitness']
        })
    
    seed_df = pd.DataFrame(seed_results)
    seed_df.to_csv('results/randomness_results.csv', index=False)
    # Calculate statistics
    best_overall_fitness = seed_df['best_fitness'].min()
    best_overall_idx = seed_df['best_fitness'].idxmin()
    best_overall_solution = seed_df.iloc[best_overall_idx]['best_solution']
    avg_fitness = seed_df['best_fitness'].mean()
    std_fitness = seed_df['best_fitness'].std()
    

    # Write summary statistics to a nicely formatted text file
    with open('results/experiment_2_seed_df.txt', 'w') as f:
        f.write("EXPERIMENT SUMMARY STATISTICS SEED DIFF.\n")
        f.write("===========================\n\n")
        f.write(f"Best solution across all seeds: {best_overall_solution}\n")
        f.write(f"Best fitness value: {best_overall_fitness}\n")
        f.write(f"Average fitness across seeds: {avg_fitness}\n")
        f.write(f"Standard deviation: {std_fitness}\n")
    
    # Second part: Rerun with decreasing population sizes
    pop_size_results = []
    
    for pop_size in population_sizes:
        seed_pop_results = []
        for seed in seeds:
            params = best_params.copy()
            params['population_size'] = pop_size
            result = run_single_experiment(params, seed=seed)
            seed_pop_results.append(result['best_fitness'])
            
        pop_size_results.append({
            'population_size': pop_size,
            'best_fitness_min': min(seed_pop_results),
            'best_fitness_avg': np.mean(seed_pop_results),
            'best_fitness_std': np.std(seed_pop_results)
        })
    
    pop_size_df = pd.DataFrame(pop_size_results)
    pop_size_df.to_csv('results/population_size_results.csv', index=False)
    
    print("\nResults with different population sizes:")
    print(pop_size_df.to_string())
    
    return seeds

def experiment_3_crossover_impact(best_params, seeds):
    """
    3. Crossover impact:
    Run the algorithm changing the crossover rate. Plot fitness across generations.
    """
    print("\n======= EXPERIMENT 3: CROSSOVER IMPACT =======")
    
    crossover_rates = [0.0,0.1,0.2,0.3, 0.5, 0.7, 0.9,1.0]
    crossover_results = {}
    
    for rate in crossover_rates:
        print(f"Testing crossover rate: {rate}")
        rate_results = []
        
        for seed in seeds:
            params = best_params.copy()
            params['crossover_rate'] = rate
            result = run_single_experiment(params, seed=seed)
            rate_results.append(result)
        
        # Average results across seeds
        avg_best_fitness = np.mean([min(r['best_fitness_history']) for r in rate_results])
        
        # Average the history arrays across seeds
        n_gens = len(rate_results[0]['best_fitness_history'])
        avg_best_history = np.zeros(n_gens)
        avg_mean_history = np.zeros(n_gens)
        
        for r in rate_results:
            avg_best_history += np.array(r['best_fitness_history'])
            avg_mean_history += np.array(r['avg_fitness_history'])
        
        avg_best_history /= len(rate_results)
        avg_mean_history /= len(rate_results)
        
        crossover_results[f"Crossover Rate {rate}"] = {
            'best_fitness': avg_best_fitness,
            'best_fitness_history': avg_best_history.tolist(),
            'avg_fitness_history': avg_mean_history.tolist()
        }
    
    # Plot the results
    plot_fitness_history(crossover_results, "Crossover Rate Comparison", "crossover_impact")
    
    # Create summary dataframe
    summary = [{
        'crossover_rate': rate,
        'avg_best_fitness': data['best_fitness']
    } for rate, data in crossover_results.items()]
    
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv('results/crossover_impact_summary.csv', index=False)
    
    print("\nCrossover Impact Summary:")
    print(summary_df.to_string())

def experiment_4_mutation_convergence(best_params, seeds):
    """
    4. Mutation and the convergence:
    Run the algorithm increasing the mutation rate and mutation strength.
    """
    print("\n======= EXPERIMENT 4: MUTATION AND CONVERGENCE =======")
    
    # Test different mutation rates and strengths
    mutation_configs = [
        {'rate': 0.0, 'strength': 0.0},
        {'rate': 0.05, 'strength': 0.05},
        {'rate': 0.1, 'strength': 0.1},
        {'rate': 0.2, 'strength': 0.2},
        {'rate': 0.3, 'strength': 0.3},
        {'rate': 0.5, 'strength': 0.5},
        {'rate': 0.7, 'strength': 0.7},
        {'rate': 0.9, 'strength': 0.9},
        {'rate': 1, 'strength': 1},
    ]
    
    mutation_results = {}
    
    for config in mutation_configs:
        config_name = f"Rate {config['rate']}, Strength {config['strength']}"
        print(f"Testing mutation config: {config_name}")
        config_results = []
        
        for seed in seeds:
            params = best_params.copy()
            params['mutation_rate'] = config['rate']
            params['mutation_strength'] = config['strength']
            result = run_single_experiment(params, seed=seed)
            config_results.append(result)
        
        # Average the results across seeds
        avg_best_fitness = np.mean([min(r['best_fitness_history']) for r in config_results])
        
        # Average the history arrays
        n_gens = len(config_results[0]['best_fitness_history'])
        avg_best_history = np.zeros(n_gens)
        avg_mean_history = np.zeros(n_gens)
        
        for r in config_results:
            avg_best_history += np.array(r['best_fitness_history'])
            avg_mean_history += np.array(r['avg_fitness_history'])
        
        avg_best_history /= len(config_results)
        avg_mean_history /= len(config_results)
        
        mutation_results[config_name] = {
            'best_fitness': avg_best_fitness,
            'best_fitness_history': avg_best_history.tolist(),
            'avg_fitness_history': avg_mean_history.tolist()
        }
    
    # Plot the results
    plot_fitness_history(mutation_results, "Mutation Parameters Comparison", "mutation_convergence")
    
    # Create summary dataframe
    summary = []
    for config_name, data in mutation_results.items():
        rate, strength = config_name.split(', ')
        rate = float(rate.split(' ')[1])
        strength = float(strength.split(' ')[1])
        
        summary.append({
            'mutation_rate': rate,
            'mutation_strength': strength,
            'avg_best_fitness': data['best_fitness']
        })
    
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv('results/mutation_convergence_summary.csv', index=False)
    
    print("\nMutation and Convergence Summary:")
    print(summary_df.to_string())

if __name__ == "__main__":
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Experiment 1: Parameter tuning
    best_params = experiment_1_parameter_tuning()
    
    # Experiment 2: Randomness analysis
    seeds = experiment_2_randomness(best_params)
    
    # Experiment 3: Crossover impact
    experiment_3_crossover_impact(best_params, seeds)
    
    # Experiment 4: Mutation and convergence
    experiment_4_mutation_convergence(best_params, seeds)
    
    print("\n======= ALL EXPERIMENTS COMPLETED =======")
    print("Results are saved in the 'results' directory.")