import networkx as nx
import numpy as np
from scipy.special import comb
import math
import numba as nb
import igraph
jit = nb.jit

def er_np(n, p, num_samples):
    for i in range(num_samples):
        yield nx.erdos_renyi_graph(n, p, directed=False)


def er_nm_approximate(n, p, num_samples):
    for i in range(num_samples):
        num_edges = np.random.binomial(comb(n, 2), p)
        yield nx.gnm_random_graph(n, num_edges, directed=False)


# Generate degree sequence given alpha, x_min for PL dist
def create_power_law_degree_sequence(n, alpha, x_min):
    while True:
        degree_sequence = [_get_plrv(alpha, x_min) for _ in range(n)]
        if not nx.is_graphical(degree_sequence):
            continue
        return degree_sequence


def create_degree_sequence_1_3(n, p_1):
    degree_sequence = []
    for i in range(n):
        if np.random.rand() < p_1:
            degree_sequence.append(1)
        else:
            degree_sequence.append(3)
    return degree_sequence


# Sample RV from discrete power law distribution
def _get_plrv(alpha, x_min):
    #np.random.seed()
    r = np.random.rand()
    #r = random.random()
    return math.floor((x_min - .5) * np.power((1 - r), (-1 / (alpha - 1))) + 0.5)


def mcmc_helper_networkx(G, num_samples, swaps_per_sample, burn=None):
    if burn:
        two_m = 2*G.number_of_edges()
        fast_mcmc_double_swap_networkx(G, two_m)
    for i in range(num_samples):
        fast_mcmc_double_swap_networkx(G, swaps_per_sample)
        yield i


# Code version of Algorithm 1 in https://arxiv.org/pdf/1608.00607.pdf
#  using stub-labels instead of vertex as there's no difference for simple graphs
#  graph space: simple, no self-loops, no multi-edges
# Direct adaptation of https://github.com/joelnish/double-edge-swap-mcmc/blob/master/dbl_edge_mcmc.py
def fast_mcmc_double_swap_networkx(G, num_swaps):
    def swap():
        # https://github.com/joelnish/double-edge-swap-mcmc/blob/master/dbl_edge_mcmc.py
        p1 = np.random.randint(num_edges)
        p2 = np.random.randint(num_edges - 1)
        if p1 == p2:  # Prevents picking the same edge twice
            p2 = num_edges - 1

        u, v = edges[p1]
        if np.random.rand() < 0.5:
            x, y = edges[p2]
        else:
            y, x = edges[p2]

        # ensure no multigraph
        if x in G[u] or y in G[v]:
            return
        if u == v and x == y:
            return

        # ensure no loops
        if u == x or u == y or v == x or v == y:
            return

        G.remove_edges_from([(u, v), (x, y)])
        G.add_edges_from([(u, x), (v, y)])
        edges[p1] = (u, x)
        edges[p2] = (v, y)

    edges = list(G.edges)
    edge_indices = list(range(G.number_of_edges()))
    num_edges = len(edge_indices)
    for i in range(num_swaps):
        swap()


def mcmc_helper_igraph(G, num_samples, swaps_per_sample, burn=None):
    edges = [e.tuple for e in G.es]
    G_nx = nx.Graph()
    G_nx.add_edges_from(edges)
    if burn:
        two_m = 2*G_nx.number_of_edges()
        fast_mcmc_double_swap_networkx(G_nx, two_m)
    for i in range(num_samples):
        fast_mcmc_double_swap_networkx(G_nx, swaps_per_sample)
        edges = [e for e in G_nx.edges]
        yield igraph.Graph(edges=edges, directed=False)
