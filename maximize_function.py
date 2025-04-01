import random
import numpy as np
import matplotlib.pyplot as plt

POPULATION_SIZE = 100
GENERATIONS = 50
MUTATION_RATE = 0.01
CROSSOVER_RATE = 0.7
ELITISM_RATE = 0.1
X_RANGE = (0, 100)
CHROMOSOME_LENGTH = 20 

# The function to maximize
def fitness_function(x):
    return -x**3 + 60*x**2 - 300*x

# Generate initial population
def initialize_population():
    return [[random.randint(0, 1) for _ in range(CHROMOSOME_LENGTH)] 
            for _ in range(POPULATION_SIZE)]

# Convert binary chromosome to real value
def decode_chromosome(chromosome):
    # Convert binary to integer
    int_value = int(''.join(map(str, chromosome)), 2)
    # Scale to x range
    max_int = 2**CHROMOSOME_LENGTH - 1
    return X_RANGE[0] + (X_RANGE[1] - X_RANGE[0]) * int_value / max_int

# Tournament selection
def selection(population, fitness_values, k=3):
    selected = []
    for _ in range(len(population)):
        # Randomly select k individuals
        candidates = random.sample(range(len(population)), k)
        # Choose the one with highest fitness
        winner = max(candidates, key=lambda x: fitness_values[x])
        selected.append(population[winner])
    return selected

# Single-point crossover
def crossover(parent1, parent2):
    if random.random() < CROSSOVER_RATE:
        point = random.randint(1, CHROMOSOME_LENGTH-1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2
    return parent1, parent2

# Bit-flip mutation
def mutation(chromosome):
    return [gene if random.random() > MUTATION_RATE else 1-gene for gene in chromosome]

# Main evolutionary algorithm
def evolutionary_algorithm():
    # Initialize population
    population = initialize_population()
    best_individual = None
    best_fitness = -float('inf')
    avg_fitness_per_generation = []
    best_fitness_per_generation = []
    
    for generation in range(GENERATIONS):
        # Decode and evaluate fitness
        decoded_pop = [decode_chromosome(ind) for ind in population]
        fitness_values = [fitness_function(x) for x in decoded_pop]
        
        # Track best solution
        current_best = max(fitness_values)
        if current_best > best_fitness:
            best_fitness = current_best
            best_index = fitness_values.index(current_best)
            best_individual = population[best_index]
        
        # Store fitness values for plotting
        avg_fitness_per_generation.append(np.mean(fitness_values))
        best_fitness_per_generation.append(best_fitness)
        
        # Selection
        selected = selection(population, fitness_values)
        
        # Crossover
        new_population = []
        for i in range(0, len(selected), 2):
            if i+1 < len(selected):
                child1, child2 = crossover(selected[i], selected[i+1])
                new_population.extend([child1, child2])
            else:
                new_population.append(selected[i])
        
        # Mutation
        new_population = [mutation(ind) for ind in new_population]
        
        # Elitism - keep best individuals
        if ELITISM_RATE > 0:
            elite_size = int(ELITISM_RATE * POPULATION_SIZE)
            elite_indices = np.argsort(fitness_values)[-elite_size:]
            for i in range(elite_size):
                new_population[i] = population[elite_indices[i]]
        
        population = new_population
        
        # Print progress
        if generation % 10 == 0:
            print(f"Generation {generation}: Best fitness = {best_fitness:.2f}")
    
    # Final result
    best_x = decode_chromosome(best_individual)
    print(f"\nOptimal solution found:")
    print(f"x = {best_x:.2f}")
    print(f"f(x) = {best_fitness:.2f}")
    
    # Plot graphs
    plt.figure(figsize=(12, 5))
    
    # Plot average fitness per generation
    plt.subplot(1, 2, 1)
    plt.plot(range(GENERATIONS), avg_fitness_per_generation, label='Média da Adaptação', color='blue')
    plt.xlabel('Geração')
    plt.ylabel('Adaptação Média')
    plt.title('Adaptação Média vs Geração')
    plt.legend()
    
    # Plot best fitness per generation
    plt.subplot(1, 2, 2)
    plt.plot(range(GENERATIONS), best_fitness_per_generation, label='Melhor Adaptação', color='red')
    plt.xlabel('Geração')
    plt.ylabel('Melhor Adaptação')
    plt.title('Melhor Indivíduo vs Geração')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return best_x, best_fitness

# Run the algorithm
evolutionary_algorithm()
