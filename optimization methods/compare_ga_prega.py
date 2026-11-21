"""
Compare a traditional GA with a PREGA-style GA on the SAME NetworkX TSP graph.

Features:
- N random 2-D cities generated with NetworkX
- Euclidean edge weights
- Tournament selection
- Order Crossover (OX)
- 2-opt segment-reversal mutation
- PREGA-style common-edge detection and common-path compression
- Best-so-far convergence histories for BOTH algorithms
- Comparison plot similar in style to the reference figure
- CSV output containing generation, GA best distance, PREGA best distance

Install:
    pip install networkx matplotlib pandas

Run:
    python compare_ga_prega.py
"""

import csv
import random
import networkx as nx
import matplotlib.pyplot as plt

N_CITIES = 200
POPULATION_SIZE = 80
GENERATIONS = 1000

CROSSOVER_PROBABILITY = 0.50
MUTATION_PROBABILITY = 0.01 
TOURNAMENT_SIZE = 3

SEED = 7
SHOW_TOUR_PLOTS = True
SAVE_CONVERGENCE_CSV = True
CSV_FILENAME = "ga_vs_prega_convergence.csv"
FIGURE_FILENAME = "ga_vs_prega_convergence.png"


def make_graph(n, seed):
    if n < 4:
        raise ValueError("N_CITIES must be at least 4.")

    rng = random.Random(seed)
    graph = nx.complete_graph(n)
    positions = {
        city: (rng.uniform(0, 100), rng.uniform(0, 100))
        for city in graph.nodes
    }
    nx.set_node_attributes(graph, positions, "pos")

    for u, v in graph.edges:
        x1, y1 = positions[u]
        x2, y2 = positions[v]
        graph[u][v]["weight"] = ((x1 - x2)**2 + (y1 - y2)**2)**0.5

    return graph


def tour_length(tour, graph):
    return sum(
        graph[tour[i]][tour[(i + 1) % len(tour)]]["weight"]
        for i in range(len(tour))
    )


def tour_edges(tour):
    return {
        frozenset((tour[i], tour[(i + 1) % len(tour)]))
        for i in range(len(tour))
    }


def nearest_neighbor_tour(graph):
    unvisited = set(graph.nodes)
    current = min(unvisited)
    unvisited.remove(current)
    tour = [current]

    while unvisited:
        nxt = min(
            unvisited,
            key=lambda city: graph[current][city]["weight"]
        )
        tour.append(nxt)
        unvisited.remove(nxt)
        current = nxt

    return tour


def make_initial_population(graph, size, seed):
    rng_state = random.getstate()
    random.seed(seed)

    population = [nearest_neighbor_tour(graph)]
    n = graph.number_of_nodes()

    while len(population) < size:
        tour = list(range(n))
        random.shuffle(tour)
        population.append(tour)

    random.setstate(rng_state)
    return [tour[:] for tour in population]


def detect_common_paths(population):

    common = tour_edges(population[0])
    for tour in population[1:]:
        common.intersection_update(tour_edges(tour))

    common_graph = nx.Graph()
    for edge in common:
        u, v = tuple(edge)
        common_graph.add_edge(u, v)

    blocks = []
    for component in nx.connected_components(common_graph):
        subgraph = common_graph.subgraph(component)
        endpoints = [v for v, degree in subgraph.degree() if degree == 1]

        # A simple path has two endpoints. Cycle components are skipped.
        if len(endpoints) != 2:
            continue

        path = [endpoints[0]]
        previous = None
        current = endpoints[0]

        while current != endpoints[1]:
            nxt = next(v for v in subgraph.neighbors(current) if v != previous)
            path.append(nxt)
            previous, current = current, nxt

        blocks.append(tuple(path))

    return blocks, common


def compress_tour(tour, blocks, common_edges):

    if not blocks:
        return [(city,) for city in tour]

    block_of_city = {
        city: block
        for block in blocks
        for city in block
    }

    n = len(tour)
    cut = next(
        (
            i for i in range(n)
            if frozenset((tour[i], tour[(i + 1) % n])) not in common_edges
        ),
        None
    )

    if cut is None:
        return [(city,) for city in tour]

    start = (cut + 1) % n
    rotated = tour[start:] + tour[:start]

    compressed = []
    i = 0

    while i < n:
        city = rotated[i]
        block = block_of_city.get(city)

        if block is None:
            compressed.append((city,))
            i += 1
            continue

        if city == block[0]:
            oriented_block = block
        elif city == block[-1]:
            oriented_block = tuple(reversed(block))
        else:
            compressed.append((city,))
            i += 1
            continue

        if tuple(rotated[i:i + len(oriented_block)]) == oriented_block:
            compressed.append(oriented_block)
            i += len(oriented_block)
        else:
            compressed.append((city,))
            i += 1

    expanded = [city for gene in compressed for city in gene]
    if len(expanded) != n or set(expanded) != set(tour):
        return [(city,) for city in tour]

    return compressed


def expand_chromosome(chromosome):
    return [city for gene in chromosome for city in gene]


def ox_crossover(parent1, parent2):

    size = len(parent1)

    if size != len(parent2):
        raise ValueError("OX parents must have equal chromosome lengths.")
    if size < 2:
        return [tuple(gene) for gene in parent1]

    left, right = sorted(random.sample(range(size), 2))
    child = [None] * size
    child[left:right + 1] = parent1[left:right + 1]

    used = {frozenset(gene) for gene in child if gene is not None}
    donor_order = parent2[right + 1:] + parent2[:right + 1]
    genes_to_insert = [
        gene for gene in donor_order
        if frozenset(gene) not in used
    ]
    empty_positions = (
        list(range(right + 1, size)) + list(range(0, left))
    )

    # Fill only the positions outside the copied segment, in OX order.
    for position, gene in zip(empty_positions, genes_to_insert):
        child[position] = gene

    # Defensive fallback if any slot remains empty.
    if any(gene is None for gene in child):
        remaining = [g for g in donor_order
                     if frozenset(g) not in used]
        for i in range(size):
            if child[i] is None and remaining:
                child[i] = remaining.pop(0)

    return child


# ----------------------------- 2-OPT MUTATION --------------------------

def two_opt_mutation(chromosome, mutation_probability):

    child = [tuple(gene) for gene in chromosome]
    size = len(child)

    if size < 2:
        return child

    event_probability = 1 - (1 - mutation_probability) ** size
    if random.random() >= event_probability:
        return child

    left, right = sorted(random.sample(range(size), 2))
    reversed_segment = [
        tuple(reversed(gene))
        for gene in reversed(child[left:right + 1])
    ]
    child[left:right + 1] = reversed_segment

    return child



def tournament_selection(population, graph, tournament_size):
    contestants = random.sample(
        population, min(tournament_size, len(population))
    )
    winner = min(contestants, key=lambda tour: tour_length(tour, graph))
    return winner[:]


def make_offspring(mating_pool, graph, population_size):
    offspring = []

    while len(offspring) < population_size:
        parent1 = random.choice(mating_pool)
        parent2 = random.choice(mating_pool)

        if random.random() < CROSSOVER_PROBABILITY:
            child1 = ox_crossover(parent1, parent2)
            child2 = ox_crossover(parent2, parent1)
        else:
            child1 = [tuple(g) for g in parent1]
            child2 = [tuple(g) for g in parent2]

        child1 = two_opt_mutation(child1, MUTATION_PROBABILITY)
        child2 = two_opt_mutation(child2, MUTATION_PROBABILITY)

        offspring.append(expand_chromosome(child1))
        if len(offspring) < population_size:
            offspring.append(expand_chromosome(child2))

    return offspring



def run_traditional_ga(graph, initial_population, seed):
    random.seed(seed)
    population = [tour[:] for tour in initial_population]

    best = min(population, key=lambda t: tour_length(t, graph))[:]
    best_distance = tour_length(best, graph)
    history = [best_distance]

    for generation in range(1, GENERATIONS + 1):
        selected = [
            tournament_selection(population, graph, TOURNAMENT_SIZE)
            for _ in range(POPULATION_SIZE)
        ]

        mating_pool = [[(city,) for city in tour] for tour in selected]
        offspring = make_offspring(mating_pool, graph, POPULATION_SIZE)

        generation_best = min(
            offspring, key=lambda t: tour_length(t, graph)
        )
        generation_distance = tour_length(generation_best, graph)

        if generation_distance < best_distance:
            best = generation_best[:]
            best_distance = generation_distance

        offspring[0] = best[:]
        population = offspring
        history.append(best_distance)

        if generation % 100 == 0:
            print(f"GA      generation {generation:4d}: {best_distance:.3f}")

    return best, best_distance, history



def run_prega(graph, initial_population, seed):
    """PREGA-style GA: CGD/CGC followed by OX and 2-opt on reduced genes."""
    random.seed(seed)
    population = [tour[:] for tour in initial_population]

    best = min(population, key=lambda t: tour_length(t, graph))[:]
    best_distance = tour_length(best, graph)
    history = [best_distance]

    for generation in range(1, GENERATIONS + 1):
        selected = [
            tournament_selection(population, graph, TOURNAMENT_SIZE)
            for _ in range(POPULATION_SIZE)
        ]

        # Pattern reduction starts after the initial generation.
        if generation >= 2:
            common_blocks, common_edges = detect_common_paths(selected)
            mating_pool = [
                compress_tour(tour, common_blocks, common_edges)
                for tour in selected
            ]
        else:
            common_blocks = []
            mating_pool = [[(city,) for city in tour] for tour in selected]

        offspring = make_offspring(mating_pool, graph, POPULATION_SIZE)

        generation_best = min(
            offspring, key=lambda t: tour_length(t, graph)
        )
        generation_distance = tour_length(generation_best, graph)

        if generation_distance < best_distance:
            best = generation_best[:]
            best_distance = generation_distance

        offspring[0] = best[:]
        population = offspring
        history.append(best_distance)

        if generation % 100 == 0:
            print(
                f"PREGA   generation {generation:4d}: {best_distance:.3f} "
                f"(common path blocks: {len(common_blocks)})"
            )

    return best, best_distance, history



def save_histories_csv(ga_history, prega_history, filename):
    with open(filename, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["generation", "GA_best_distance", "PREGA_best_distance"])
        for generation, (ga_value, prega_value) in enumerate(
            zip(ga_history, prega_history)
        ):
            writer.writerow([generation, ga_value, prega_value])
    print(f"Saved convergence data to: {filename}")


def plot_convergence(ga_history, prega_history):
    """Plot both best-so-far curves on the same axes."""
    generations = range(len(ga_history))

    plt.figure(figsize=(10, 6))
    plt.plot(generations, ga_history, label="Traditional GA", color="red")
    plt.plot(generations, prega_history, label="PREGA", color="blue",
             linestyle="--")

    plt.xlabel("Generation")
    plt.ylabel("Best-so-far tour distance")
    plt.title(f"Traditional GA vs PREGA (N = {N_CITIES})")
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_FILENAME, dpi=300)
    plt.show()
    print(f"Saved convergence plot to: {FIGURE_FILENAME}")


def plot_best_tour(graph, tour, title):
    pos = nx.get_node_attributes(graph, "pos")
    route_edges = list(zip(tour, tour[1:] + tour[:1]))

    plt.figure(figsize=(7, 6))
    nx.draw_networkx_nodes(graph, pos, node_size=35, node_color="skyblue")
    nx.draw_networkx_edges(
        graph, pos, edgelist=route_edges, edge_color="tomato", width=1.5
    )
    nx.draw_networkx_labels(graph, pos, font_size=6)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()



def main():
    graph = make_graph(N_CITIES, SEED)
    initial_population = make_initial_population(
        graph, POPULATION_SIZE, SEED + 1
    )

    print(f"TSP graph: {N_CITIES} cities, {graph.number_of_edges()} edges")
    print("\nRunning traditional GA...")
    ga_tour, ga_distance, ga_history = run_traditional_ga(
        graph, initial_population, SEED + 10
    )

    print("\nRunning PREGA")
    prega_tour, prega_distance, prega_history = run_prega(
        graph, initial_population, SEED + 20
    )
    print()

    print(f"Traditional GA best distance: {ga_distance:.3f}")
    print(f"PREGA best distance:          {prega_distance:.3f}")

    improvement = (ga_distance - prega_distance) / ga_distance * 100
    print(f"PREGA distance improvement vs GA: {improvement:.2f}%")
    print("Note: a negative value means GA found the shorter tour.")

    if SAVE_CONVERGENCE_CSV:
        save_histories_csv(ga_history, prega_history, CSV_FILENAME)

    plot_convergence(ga_history, prega_history)

    if SHOW_TOUR_PLOTS:
        plot_best_tour(graph, ga_tour, "Best traditional GA tour")
        plot_best_tour(graph, prega_tour, "Best PREGA tour")


if __name__ == "__main__":
    main()
