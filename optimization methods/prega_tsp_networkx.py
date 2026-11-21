"""
PREGA-style TSP Genetic Algorithm using NetworkX.

Edit N_CITIES and the GA parameters below.
Dependencies: pip install networkx matplotlib
"""

import random
import networkx as nx
import matplotlib.pyplot as plt

# ---------------- USER PARAMETERS ----------------
N_CITIES = 30
POPULATION_SIZE = 80
GENERATIONS = 300
CROSSOVER_PROBABILITY = 0.50
MUTATION_PROBABILITY = 0.01  # per compressed gene
TOURNAMENT_SIZE = 3
SEED = 7
SHOW_PLOTS = True
# -------------------------------------------------

def make_graph(n, seed):
    """Random 2-D cities with Euclidean distances on a complete graph."""
    if n < 4:
        raise ValueError("N_CITIES must be at least 4.")
    rng = random.Random(seed)
    G = nx.complete_graph(n)
    pos = {v: (rng.random() * 100, rng.random() * 100) for v in G}
    nx.set_node_attributes(G, pos, "pos")
    for u, v in G.edges:
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        G[u][v]["weight"] = ((x1-x2)**2 + (y1-y2)**2)**0.5
    return G

def distance(tour, G):
    return sum(G[tour[i]][tour[(i+1) % len(tour)]]["weight"]
               for i in range(len(tour)))

def edges_of(tour):
    return {frozenset((tour[i], tour[(i+1) % len(tour)]))
            for i in range(len(tour))}

def initial_population(n, size, G):
    # Include one nearest-neighbour tour, then random permutations.
    unvisited = set(range(n))
    nn = [0]
    unvisited.remove(0)
    while unvisited:
        nxt = min(unvisited, key=lambda x: G[nn[-1]][x]["weight"])
        nn.append(nxt)
        unvisited.remove(nxt)
    pop = [nn]
    for _ in range(size - 1):
        t = list(range(n))
        random.shuffle(t)
        pop.append(t)
    return pop

# A gene is a tuple of cities: (city,) for a normal gene or
# (city_a, city_b, ...) for a compressed common path.
def detect_common_paths(population):
    """CGD: find edges present in every selected tour; return path blocks."""
    common = edges_of(population[0])
    for tour in population[1:]:
        common.intersection_update(edges_of(tour))

    H = nx.Graph()
    for edge in common:
        u, v = tuple(edge)
        H.add_edge(u, v)

    blocks = []
    for component in nx.connected_components(H):
        S = H.subgraph(component)
        ends = [v for v, deg in S.degree() if deg == 1]
        # A simple path has exactly two endpoints. Skip cycle components.
        if len(ends) != 2:
            continue
        path, prev, cur = [ends[0]], None, ends[0]
        while cur != ends[1]:
            nxt = next(v for v in S.neighbors(cur) if v != prev)
            path.append(nxt)
            prev, cur = cur, nxt
        blocks.append(tuple(path))
    return blocks, common

def compress_tour(tour, blocks, common_edges):
    """CGC: replace each maximal common path by one composite gene."""
    if not blocks:
        return [(v,) for v in tour]

    block_of = {v: b for b in blocks for v in b}
    n = len(tour)

    # Rotate after a non-common edge, so a common path is not split at ends.
    cut = next((i for i in range(n)
                if frozenset((tour[i], tour[(i+1) % n])) not in common_edges),
               None)
    if cut is None:
        return [(v,) for v in tour]
    start = (cut + 1) % n
    seq = tour[start:] + tour[:start]

    result, i = [], 0
    while i < n:
        v = seq[i]
        b = block_of.get(v)
        if b is None:
            result.append((v,))
            i += 1
            continue
        oriented = b if v == b[0] else tuple(reversed(b)) if v == b[-1] else None
        if oriented is not None and tuple(seq[i:i+len(b)]) == oriented:
            result.append(oriented)
            i += len(b)
        else:
            result.append((v,))
            i += 1

    # Safety check: compression must preserve every city exactly once.
    flat = [v for gene in result for v in gene]
    if len(flat) != n or set(flat) != set(tour):
        return [(v,) for v in tour]
    return result

def expand(chromosome):
    return [city for gene in chromosome for city in gene]

def ox_crossover(p1, p2):
    """Order Crossover on genes; a composite path is one indivisible item."""
    n = len(p1)
    if n < 2:
        return [tuple(g) for g in p1]
    a, b = sorted(random.sample(range(n), 2))
    child = [None] * n
    child[a:b+1] = p1[a:b+1]
    used = {frozenset(g) for g in child if g is not None}
    donor = p2[b+1:] + p2[:b+1]
    insert = [g for g in donor if frozenset(g) not in used]
    positions = list(range(b+1, n)) + list(range(0, b+1))
    for pos, gene in zip(positions, insert):
        child[pos] = gene
    return child

def two_opt_mutation(chromosome, p_mut):
    """
    Standard 2-opt mutation: reverse a contiguous section of genes.
    Reverse each composite gene internally as well, preserving its path.
    """
    child = [tuple(g) for g in chromosome]
    n = len(child)
    # Per-gene probability p_mut: trigger if at least one gene is selected.
    if n < 2 or random.random() >= 1 - (1-p_mut)**n:
        return child
    i, j = sorted(random.sample(range(n), 2))
    child[i:j+1] = [tuple(reversed(g)) for g in reversed(child[i:j+1])]
    return child

def tournament(pop, G, k):
    candidates = random.sample(pop, min(k, len(pop)))
    return min(candidates, key=lambda t: distance(t, G))[:]

def run_prega(G):
    n = G.number_of_nodes()
    pop = initial_population(n, POPULATION_SIZE, G)
    best = min(pop, key=lambda t: distance(t, G))[:]
    best_d = distance(best, G)
    history = []

    for gen in range(1, GENERATIONS + 1):
        # Selection
        selected = [tournament(pop, G, TOURNAMENT_SIZE)
                    for _ in range(POPULATION_SIZE)]

        # PREGA pattern reduction starts from generation 2, as in the paper.
        if gen >= 2:
            blocks, common_edges = detect_common_paths(selected)
            mating = [compress_tour(t, blocks, common_edges) for t in selected]
        else:
            blocks = []
            mating = [[(v,) for v in t] for t in selected]

        # Elitism: preserve the best full tour.
        new_pop = [best[:]]
        while len(new_pop) < POPULATION_SIZE:
            p1, p2 = random.choice(mating), random.choice(mating)
            if random.random() < CROSSOVER_PROBABILITY:
                c1, c2 = ox_crossover(p1, p2), ox_crossover(p2, p1)
            else:
                c1, c2 = [tuple(g) for g in p1], [tuple(g) for g in p2]

            c1 = two_opt_mutation(c1, MUTATION_PROBABILITY)
            c2 = two_opt_mutation(c2, MUTATION_PROBABILITY)
            new_pop.append(expand(c1))
            if len(new_pop) < POPULATION_SIZE:
                new_pop.append(expand(c2))

        pop = new_pop
        candidate = min(pop, key=lambda t: distance(t, G))
        candidate_d = distance(candidate, G)
        if candidate_d < best_d:
            best, best_d = candidate[:], candidate_d
        history.append(best_d)

        if gen == 1 or gen % 25 == 0:
            print(f"Generation {gen:4d} | best distance = {best_d:.3f} "
                  f"| common path blocks = {len(blocks)}")

    return best, best_d, history

def show_results(G, best, history):
    pos = nx.get_node_attributes(G, "pos")
    route = list(zip(best, best[1:] + best[:1]))
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))

    nx.draw_networkx_nodes(G, pos, ax=ax[0], node_size=45,
                           node_color="skyblue")
    nx.draw_networkx_edges(G, pos, ax=ax[0], edgelist=route,
                           edge_color="tomato", width=1.8)
    nx.draw_networkx_labels(G, pos, ax=ax[0], font_size=7)
    ax[0].set_title("Best TSP tour")
    ax[0].axis("off")

    ax[1].plot(history)
    ax[1].set_title("Best distance by generation")
    ax[1].set_xlabel("Generation")
    ax[1].set_ylabel("Tour distance")
    ax[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    random.seed(SEED)
    graph = make_graph(N_CITIES, SEED)
    print(f"Created complete graph: {N_CITIES} cities, {graph.number_of_edges()} edges")

    best_tour, best_distance, convergence = run_prega(graph)

    print("\nBest tour:", best_tour)
    print(f"Best tour length: {best_distance:.3f}")

    if SHOW_PLOTS:
        show_results(graph, best_tour, convergence)
