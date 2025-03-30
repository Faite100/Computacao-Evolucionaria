import random
import matplotlib.pyplot as plt
import networkx as nx

# gera cidade aleatória
def generate_graph(num_nodes):
    graph = {i: {} for i in range(num_nodes)}
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            weight = random.randint(1, 100)
            graph[i][j] = weight
            graph[j][i] = weight
    return graph

# população inicial de caminhos aleatórios
def create_population(graph, size):
    nodes = list(graph.keys())
    return [random.sample(nodes, len(nodes)) for _ in range(size)]

# calcula a distância total do caminho
def fitness(graph, path):
    distance = sum(graph[path[i]][path[i+1]] for i in range(len(path) - 1))
    distance += graph[path[-1]][path[0]]
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
    
    for i in range(generations):
        new_population = []
        for j in range(pop_size // 2):
            parent1 = tournament_selection(graph, population)
            parent2 = tournament_selection(graph, population)
            child1, child2 = ordered_crossover(parent1, parent2), ordered_crossover(parent2, parent1)
            if random.random() < mutation_rate:
                swap_mutation(child1)
            if random.random() < mutation_rate:
                swap_mutation(child2)
            new_population.extend([child1, child2])
        
        population = new_population
        current_best = min(population, key=lambda p: fitness(graph, p))
        current_best_distance = fitness(graph, current_best)
        if current_best_distance < best_distance:
            best_solution, best_distance = current_best, current_best_distance
    
    return best_solution, best_distance

# plota o grafo original e o melhor caminho encontrado
def plot_graph(graph, best_path):
    G = nx.Graph()
    for node, edges in graph.items():
        for neighbor, weight in edges.items():
            G.add_edge(node, neighbor, weight=weight)
    pos = nx.spring_layout(G, seed=42)
    
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.title("original")
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray')
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    
    plt.subplot(1, 2, 2)
    plt.title("melhor caminho encontrado")
    edges = [(best_path[i], best_path[i+1]) for i in range(len(best_path)-1)] + [(best_path[-1], best_path[0])]
    nx.draw(G, pos, with_labels=True, node_color='lightgreen', edge_color='gray')
    nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color='red', width=2)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    
    plt.show()

n_city = 5
random_city = generate_graph(n_city)
t1 = {0: {1: 76, 2: 48, 3: 89, 4: 66}, 1: {0: 76, 2: 77, 3: 97, 4: 79}, 2: {0: 48, 1: 77, 3: 18, 4: 95}, 3: {0: 89, 1: 97, 2: 18, 4: 86}, 4: {0: 66, 1: 79, 2: 95, 3: 86}}
best_path, best_distance = genetic_algorithm(random_city)
print(f"Melhor caminho encontrado: {best_path} com distância total de {best_distance} unidades")
plot_graph(random_city, best_path)