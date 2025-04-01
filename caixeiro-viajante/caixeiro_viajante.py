import random
import matplotlib.pyplot as plt
import networkx as nx
import math

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
        for j in range(pop_size // 2):
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

    # Janela 2: Gráfico de Adaptação Média
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

best_path, best_distance, avg_distances, best_distances = genetic_algorithm(random_city)
print(f"Melhor caminho encontrado: {best_path} com distância total de {best_distance} unidades")
plot_graph(random_city, best_path, avg_distances, best_distances, best_distance)
# ----------------------------- #