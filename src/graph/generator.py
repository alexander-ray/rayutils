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
