import networkx as nx
import numpy as np
from scipy.special import comb
import math
import numba as nb
import igraph
jit = nb.jit


def er_np(n, p, num_samples):
    """
    Generator, yields num_samples G(n,p) graphs with networkx.

    :param n: Number of nodes
    :param p: Probability of connection between nodes
    :param num_samples: Number of graphs
    """
    for i in range(num_samples):
        yield nx.erdos_renyi_graph(n, p, directed=False)


def er_nm_approximate(n, p, num_samples):
    """
    Generator, yields num_samples G(n,p) graphs with networkx.
    Approximation using binomial theorem to exploit speed of networkx gnm_random_graph.

    :param n: Number of nodes
    :param p: Probability of connection between nodes
    :param num_samples: Number of graphs
    """
    for i in range(num_samples):
        num_edges = np.random.binomial(comb(n, 2), p)
        yield nx.gnm_random_graph(n, num_edges, directed=False)


# Generate degree sequence given alpha, x_min for PL dist
def create_power_law_degree_sequence(n, alpha, x_min):
    """
    Helper method to create a graphical (as defined by networkx) power law degree sequence.
    Somewhat unhelpful--look for nx or igraph versions

    :param n: Number of nodes
    :param alpha: Alpha param for power law
    :param x_min: x_min param for power law
    :return: Returns list of degrees
    """
    # Sample RV from discrete power law distribution
    def get_plrv(alpha, xmin):
        # np.random.seed()
        r = np.random.rand()
        # r = random.random()
        return math.floor((x_min - .5) * np.power((1 - r), (-1 / (alpha - 1))) + 0.5)

    while True:
        degree_sequence = [get_plrv(alpha, x_min) for _ in range(n)]
        if not nx.is_graphical(degree_sequence):
            continue
        return degree_sequence


def mcmc_helper_networkx(G, num_samples, swaps_per_sample, burn=None):
    """
    Driver function for MCMC double edge swap on a networkx graph.
    Generator that yields nx.Graph objects

    :param G: nx.Graph
    :param num_samples: Number of graph snapshots to yield
    :param swaps_per_sample: Number of double edge swaps between samples
    :param burn: Truthy variable to include burn-in of 2m swaps
    """
    if burn:
        two_m = 2*G.number_of_edges()
        fast_mcmc_double_swap_networkx(G, two_m)
    for i in range(num_samples):
        fast_mcmc_double_swap_networkx(G, swaps_per_sample)
        yield i


def mcmc_helper_igraph(G, num_samples, swaps_per_sample, burn=None):
    """
    Driver function for MCMC double edge swap on an igraph.Graph.
    Converts to nx.Graph to do swaps, as the lack of constant-time edge swaps in igraph is prohibitively slow.
    Generator that yields igraph.Graph objects

    :param G: igraph.Graph
    :param num_samples: Number of graph snapshots to yield
    :param swaps_per_sample: Number of double edge swaps between samples
    :param burn: Truthy variable to include burn-in of 2m swaps
    """
    edges = [e.tuple for e in G.es]
    G_nx = nx.Graph()
    G_nx.add_edges_from(edges)
    if burn:
        two_m = 2 * G_nx.number_of_edges()
        fast_mcmc_double_swap_networkx(G_nx, two_m)
    for i in range(num_samples):
        fast_mcmc_double_swap_networkx(G_nx, swaps_per_sample)
        edges = [e for e in G_nx.edges]
        yield igraph.Graph(edges=edges, directed=False)


def fast_mcmc_double_swap_networkx(G, num_swaps):
    """
    Code version of Algorithm 1 in https://arxiv.org/pdf/1608.00607.pdf
     using stub-labels instead of vertex as there's no difference for simple graphs
     graph space: simple, no self-loops, no multi-edges
    Direct adaptation of https://github.com/joelnish/double-edge-swap-mcmc/blob/master/dbl_edge_mcmc.py

    :param G: nx.Graph
    :param num_swaps: Number of swap attempts
    :return: None
    """

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


