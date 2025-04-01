import random
import matplotlib.pyplot as plt
import networkx as nx


def generate(num_nodes, edge_prob=0.5):
    graph = {i: {} for i in range(num_nodes)}
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if random.random() < edge_prob:  # probabilidade de criar uma aresta entre os nós
                weight = random.randint(1, 100)
                graph[i][j] = weight
                graph[j][i] = weight
    return graph

# cria população inicial de caminhos válidos
def create_population(graph, size):
    nodes = list(graph.keys())
    valid_paths = []
    while len(valid_paths) < size:
        # print(f"tentativa {len(valid_paths)}") 
        path = random.sample(nodes, len(nodes))
        if all(path[i+1] in graph[path[i]] for i in range(len(path) - 1)) and path[0] in graph[path[-1]]:
            valid_paths.append(path)
    # print(valid_paths)
    return valid_paths

# calcula a distância total do caminho e penaliza caminhos inválidos
def fitness(graph, path):
    distance = 0
    # penalização por caminhos inválidos
    for i in range(len(path) - 1):
        if path[i+1] not in graph[path[i]]:
            # penaliza com uma grande quantidade proporcional à posição no caminho
            distance += 1000  # penalização constante para caminhos inválidos
        else:
            distance += graph[path[i]][path[i+1]]

    if path[0] not in graph[path[-1]]:
        distance += 1000  # penalização para não completar o ciclo
    
    return distance

# seleção por torneio - escolhe k indivíduos aleatórios e retorna o melhor
def tournament_selection(graph, population, k=3):
    selected = random.sample(population, k)
    best = selected[0]
    for c in selected:
        if fitness(graph, c) < fitness(graph, best):
            best = c
    return best

# crossover por ordenação (OX)
def ordered_crossover(parent1, parent2):
    size = len(parent1)
    a, b = sorted(random.sample(range(size), 2))
    child = [None] * size
    child[a:b] = parent1[a:b]
    remaining = [gene for gene in parent2 if gene not in child]
    index = 0
    for i in range(size):
        if child[i] is None:
            child[i] = remaining[index]
            index += 1
    return child

# mutação de troca de posições
def swap_mutation(individual):
    a, b = random.sample(range(len(individual)), 2)
    individual[a], individual[b] = individual[b], individual[a]

def genetic_algorithm(graph, pop_size=50, generations=500, mutation_rate=0.1):
    population = create_population(graph, pop_size)
    best_solution = min(population, key=lambda p: fitness(graph, p))
    best_distance = fitness(graph, best_solution)

    best_distances = []
    avg_distances = []

    for i in range(generations):
        new_population = []

        # mantém as melhores soluções
        new_population.append(best_solution)
        
        while len(new_population) < pop_size:
            parent1 = tournament_selection(graph, population)
            parent2 = tournament_selection(graph, population)
            child1 = ordered_crossover(parent1, parent2)
            child2 = ordered_crossover(parent2, parent1)
            if random.random() < mutation_rate:
                swap_mutation(child1)
            if random.random() < mutation_rate:
                swap_mutation(child2)
            new_population.extend([child1, child2])

        population = new_population

        fitness_values = [fitness(graph, p) for p in population]
        avg_distance = sum(fitness_values) / len(fitness_values)
        current_best = min(population, key=lambda p: fitness(graph, p))
        current_best_distance = fitness(graph, current_best)

        if current_best_distance < best_distance:
            best_solution = current_best
            best_distance = current_best_distance

        best_distances.append(current_best_distance)
        avg_distances.append(avg_distance)

    return best_solution, best_distance, avg_distances, best_distances

def plot_graph(graph, best_path, avg_distances, best_distances, best_distance):
    G = nx.Graph()
    for node, edges in graph.items():
        for neighbor, weight in edges.items():
            G.add_edge(node, neighbor, weight=weight)
    pos = nx.spring_layout(G, seed=42)

    plt.figure(figsize=(12, 8))
    plt.title("melhor caminho encontrado")
    edges = [(best_path[i], best_path[i+1]) for i in range(len(best_path)-1)] + [(best_path[-1], best_path[0])]
    nx.draw(G, pos, with_labels=True, node_color='lightgreen', edge_color='gray')
    nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color='red', width=2)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

    plt.text(0.05, 0.95, f"melhor distância: {best_distance}\nordem de visitação: {best_path}",
            transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.show()

    plt.figure(figsize=(10, 7))
    plt.plot(avg_distances, label='adaptação média', color='blue')
    plt.title("evolução da adaptação média")
    plt.xlabel("Geração")
    plt.ylabel("Distância")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 7))
    plt.plot(best_distances, label='melhor indivíduo', color='green')
    plt.title("evolução do melhor indivíduo")
    plt.xlabel("Geração")
    plt.ylabel("Distância")
    plt.legend()
    plt.grid(True)
    plt.show()

# ----------------------------- #
n_city = 15
random_city = generate(n_city, edge_prob=0.75)
# random_city = {0: {1: 54, 2: 36, 3: 9, 5: 12, 6: 23, 8: 90, 11: 74, 12: 94, 13: 11, 16: 51, 18: 37, 19: 47, 21: 34, 22: 20, 23: 71, 24: 86, 25: 2, 26: 92, 29: 79}, 1: {0: 54, 2: 100, 4: 70, 6: 75, 7: 22, 13: 28, 14: 92, 15: 79, 16: 12, 19: 99, 20: 92, 21: 44, 22: 53, 23: 88, 24: 55, 25: 7, 26: 56, 27: 23, 28: 38, 29: 48}, 2: {0: 36, 1: 100, 3: 20, 4: 84, 5: 23, 6: 41, 7: 48, 8: 38, 9: 54, 10: 7, 11: 22, 12: 70, 13: 46, 14: 7, 15: 53, 17: 56, 18: 55, 19: 100, 20: 33, 21: 13, 22: 94, 23: 52, 24: 58, 25: 27, 27: 23, 28: 58, 29: 12}, 3: {0: 9, 2: 20, 4: 26, 5: 6, 7: 54, 8: 54, 10: 87, 11: 30, 12: 13, 13: 61, 15: 33, 17: 18, 18: 99, 19: 73, 20: 96, 21: 57, 22: 98, 23: 42, 24: 70, 27: 33, 28: 96}, 4: {1: 70, 2: 84, 3: 26, 5: 60, 6: 23, 7: 100, 8: 68, 9: 12, 10: 54, 11: 81, 12: 31, 13: 41, 15: 25, 16: 47, 17: 46, 18: 79, 19: 9, 20: 90, 21: 74, 22: 54, 23: 85, 24: 27, 26: 4, 27: 36, 28: 87, 29: 29}, 5: {0: 12, 2: 23, 3: 6, 4: 60, 6: 46, 7: 40, 8: 43, 9: 55, 11: 41, 12: 41, 13: 54, 14: 7, 17: 31, 20: 72, 21: 11, 22: 6, 23: 99, 24: 19, 25: 45, 28: 8, 29: 78}, 6: {0: 23, 1: 75, 2: 41, 4: 23, 5: 46, 7: 45, 8: 93, 11: 37, 12: 89, 13: 53, 17: 7, 18: 3, 19: 30, 20: 58, 21: 48, 22: 1, 23: 26, 24: 57, 25: 25, 26: 52, 27: 97, 28: 50, 29: 100}, 7: {1: 22, 2: 48, 3: 54, 4: 100, 5: 40, 6: 45, 9: 91, 10: 55, 11: 19, 12: 51, 13: 40, 14: 86, 17: 84, 18: 15, 23: 85, 25: 98, 26: 63, 27: 82, 28: 55, 29: 13}, 8: {0: 90, 2: 38, 3: 54, 4: 68, 5: 43, 6: 93, 9: 85, 10: 61, 11: 86, 12: 47, 13: 5, 14: 12, 16: 61, 18: 31, 21: 70, 22: 77, 23: 54, 25: 93, 26: 53, 27: 1, 28: 82, 29: 84}, 9: {2: 54, 4: 12, 5: 55, 7: 91, 8: 85, 13: 50, 16: 94, 17: 69, 18: 47, 20: 90, 22: 39, 23: 87, 27: 33, 28: 56, 29: 91}, 10: {2: 7, 3: 87, 4: 54, 7: 55, 8: 61, 12: 80, 13: 87, 14: 49, 17: 71, 18: 70, 19: 83, 20: 22, 21: 94, 22: 9, 23: 81, 24: 48, 25: 59, 26: 46, 27: 35, 28: 79}, 11: {0: 74, 2: 22, 3: 30, 4: 81, 5: 41, 6: 37, 7: 19, 8: 86, 12: 58, 14: 52, 15: 18, 16: 90, 17: 73, 18: 64, 19: 89, 20: 52, 21: 78, 23: 97, 24: 99, 25: 61, 26: 38, 27: 82, 28: 98, 29: 4}, 12: {0: 94, 2: 70, 3: 13, 4: 31, 5: 41, 6: 89, 7: 51, 8: 47, 10: 80, 11: 58, 13: 58, 14: 74, 15: 14, 16: 62, 18: 88, 19: 1, 20: 79, 21: 76, 22: 90, 23: 59, 24: 67, 25: 78, 26: 71, 27: 46, 29: 48}, 13: {0: 11, 1: 28, 2: 46, 3: 61, 4: 41, 5: 54, 6: 53, 7: 40, 8: 5, 9: 50, 10: 87, 12: 58, 14: 80, 15: 41, 16: 86, 17: 86, 18: 54, 19: 30, 20: 11, 21: 60, 23: 45, 24: 22, 26: 38, 27: 25, 28: 4}, 14: {1: 92, 2: 7, 5: 7, 7: 86, 8: 12, 10: 49, 11: 52, 12: 74, 13: 80, 15: 2, 16: 95, 17: 93, 21: 30, 22: 53, 24: 33, 27: 44, 28: 25, 29: 72}, 15: {1: 79, 2: 53, 3: 33, 4: 25, 11: 18, 12: 14, 13: 41, 14: 2, 16: 25, 17: 23, 18: 69, 19: 71, 20: 42, 21: 97, 23: 63, 24: 91, 25: 6, 26: 23, 27: 97, 28: 42, 29: 57}, 16: {0: 51, 1: 12, 4: 47, 8: 61, 9: 94, 11: 90, 12: 62, 13: 86, 14: 95, 15: 25, 17: 47, 18: 27, 19: 2, 21: 24, 22: 12, 25: 37, 26: 86, 27: 75, 29: 74}, 17: {2: 56, 3: 18, 4: 46, 5: 31, 6: 7, 7: 84, 9: 69, 10: 71, 11: 73, 13: 86, 14: 93, 15: 23, 16: 47, 18: 67, 19: 82, 20: 96, 21: 16, 22: 24, 25: 95, 27: 40, 28: 14}, 18: {0: 37, 2: 55, 3: 99, 4: 79, 6: 3, 7: 15, 8: 31, 9: 47, 10: 70, 11: 64, 12: 88, 13: 54, 15: 69, 16: 27, 17: 67, 21: 15, 22: 15, 23: 11, 24: 15, 26: 97, 27: 38, 28: 31, 29: 5}, 19: {0: 47, 1: 99, 2: 100, 3: 73, 4: 9, 6: 30, 10: 83, 11: 89, 12: 1, 13: 30, 15: 71, 16: 2, 17: 82, 20: 68, 21: 10, 22: 64, 26: 72, 28: 34}, 20: {1: 92, 2: 33, 3: 96, 4: 90, 5: 72, 6: 58, 9: 90, 10: 22, 11: 52, 12: 79, 13: 11, 15: 42, 17: 96, 19: 68, 21: 20, 22: 27, 25: 84, 26: 91, 27: 69, 28: 51, 29: 73}, 21: {0: 34, 1: 44, 2: 13, 3: 57, 4: 74, 5: 11, 6: 48, 8: 70, 10: 94, 11: 78, 12: 76, 13: 60, 14: 30, 15: 97, 16: 24, 17: 16, 18: 15, 19: 10, 20: 20, 22: 66, 23: 59, 25: 2, 26: 33, 27: 47, 28: 76, 29: 46}, 22: {0: 20, 1: 53, 2: 94, 3: 98, 4: 54, 5: 6, 6: 1, 8: 77, 9: 39, 10: 9, 12: 90, 14: 53, 16: 12, 17: 24, 18: 15, 19: 64, 20: 27, 21: 66, 23: 45, 25: 46, 26: 92, 27: 59, 28: 60, 29: 43}, 23: {0: 71, 1: 88, 2: 52, 3: 42, 4: 85, 5: 99, 6: 26, 7: 85, 8: 54, 9: 87, 10: 81, 11: 97, 12: 59, 13: 45, 15: 63, 18: 11, 21: 59, 22: 45, 24: 38, 26: 76, 28: 30}, 24: {0: 86, 1: 55, 2: 58, 3: 70, 4: 27, 5: 19, 6: 57, 10: 48, 11: 99, 12: 67, 13: 22, 14: 33, 15: 91, 18: 15, 23: 38, 25: 64, 26: 100, 27: 93, 29: 19}, 25: {0: 2, 1: 7, 2: 27, 5: 45, 6: 25, 7: 98, 8: 93, 10: 59, 11: 61, 12: 78, 15: 6, 16: 37, 17: 95, 20: 84, 21: 2, 22: 46, 24: 64, 26: 13, 28: 89, 29: 49}, 26: {0: 92, 1: 56, 4: 4, 6: 52, 7: 63, 8: 53, 10: 46, 11: 38, 12: 71, 13: 38, 15: 23, 16: 86, 18: 97, 19: 72, 20: 91, 21: 33, 22: 92, 23: 76, 24: 100, 25: 13, 27: 44, 28: 5, 29: 23}, 27: {1: 23, 2: 23, 3: 33, 4: 36, 6: 97, 7: 82, 8: 1, 9: 33, 10: 35, 11: 82, 12: 46, 13: 25, 14: 44, 15: 97, 16: 75, 17: 40, 18: 38, 20: 69, 21: 47, 22: 59, 24: 93, 26: 44, 29: 71}, 28: {1: 38, 2: 58, 3: 96, 4: 87, 5: 8, 6: 50, 7: 55, 8: 82, 9: 56, 10: 79, 11: 98, 13: 4, 14: 25, 15: 42, 17: 14, 18: 31, 19: 34, 20: 51, 21: 76, 22: 60, 23: 30, 25: 89, 26: 5, 29: 92}, 29: {0: 79, 1: 48, 2: 12, 4: 29, 5: 78, 6: 100, 7: 13, 8: 84, 9: 91, 11: 4, 12: 48, 14: 72, 15: 57, 16: 74, 18: 5, 20: 73, 21: 46, 22: 43, 24: 19, 25: 49, 26: 23, 27: 71, 28: 92}}

best_path, best_distance, avg_distances, best_distances = genetic_algorithm(random_city)
print(f"Melhor caminho encontrado: {best_path} com distância total de {best_distance} unidades")
plot_graph(random_city, best_path, avg_distances, best_distances, best_distance)
# ----------------------------- #