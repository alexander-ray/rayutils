import networkx as nx
import numpy as np
from collections import Counter
from .bidirectional_bfs import bidirectional_bfs_distance_networkx, bidirectional_bfs_distance_igraph


def sampler(G, num_observations):
    """
    Rejection sampler for pairwise distances using nx.Graph.
    Too slow for use with disconnected graphs, most of the time.

    :param G: nx.Graph
    :param num_observations: Number of successful samples
    :return: Average pairwise distance
    """
    counter = 0
    for x in range(num_observations):
        i, j = np.random.choice(G.nodes, 2)
        while True:
            try:
                sp = nx.shortest_path_length(G, source=i, target=j, weight=None)
                counter += sp
            except nx.NetworkXNoPath:
                continue
            break
    return counter / num_observations


def sampler_no_rejection(G, num_samples):
    """
    "No-rejection" sampler for pairwise distances using nx.Graph.
    Chooses connected component i with probability proportional to n_i^2.
    Samples pairwise distance within selected component.

    :param G: nx.Graph
    :param num_samples: Number of samples
    :return: List of pairwise distances
    """
    # Return early if graph too small
    if len(G) == 0 or len(G) == 1:
        return

    tracker = []

    # Get list of set of nodes for each connected component
    components = list(c for c in nx.connected_components(G))
    # Make probabilities proportional to n_i ** 2
    tmp = [(len(g)**2) for g in components]
    probabilities = [n / sum(tmp) for n in tmp]
    for x in range(num_samples):
        # Get subgraph, choosing with p=((n_i choose 2)/sum of all n choose 2)
        subgraph = G.subgraph(np.random.choice(components, p=probabilities))

        # If array is of size 1, will return the single node twice
        i, j = np.random.choice(list(subgraph.nodes), 2, replace=True)
        tracker.append(nx.shortest_path_length(subgraph, source=i, target=j, weight=None))
        #tracker.append(bidirectional_bfs_distance_networkx(subgraph, s=i, t=j))
    return tracker


def ew_sampler_igraph(g, alpha=0.05, t=0.01):
    """
    :param G: igraph.Graph
    :param alpha: Confidence level
    :param t: Precision level
    :return: Dict of proportions
    """
    num_obs = int(np.ceil(np.log(2 / alpha) / (2 * (t ** 2))))
    s = sampler_no_rejection_igraph(g, num_obs)
    return {k: v/len(s) for k, v in Counter(s).items()}


def threshold_sampler_igraph(g, threshold=0.1, batch_size=1000):
    """
    :param G: igraph.Graph
    :param threshold: Threshold value
    :param batch_size: Number of samples to take before re-evaluating
    :return: List of samples
    """
    s1 = []
    s2 = []
    half_batch = int(batch_size/2)
    threshold_met = False
    while not threshold_met:
        s1.extend(sampler_no_rejection_igraph(g, half_batch))
        s2.extend(sampler_no_rejection_igraph(g, half_batch))
        if np.abs(np.mean(s1) - np.mean(s2)) < threshold:
            threshold_met = True
    return s1 + s2


def sampler_no_rejection_igraph(g, num_samples):
    """
    igraph version of "no-rejection" sampler for pairwise distances.
    Chooses connected component i with probability proportional to n_i^2.
    Samples pairwise distance within selected component.

    :param g: igraph.Graph
    :param num_samples: Number of samples
    :return: List of pairwise distances
    """
    # Return early if graph too small
    if g.vcount() == 0 or g.vcount() == 1:
        return

    tracker = []
    components, probabilities = _component_probability_generator_igraph(g)
    num_components = len(components)
    for x in range(num_samples):
        subgraph_index = np.random.choice(num_components, p=probabilities)
        i, j = np.random.choice(components[subgraph_index], 2, replace=True)
        tracker.append(bidirectional_bfs_distance_igraph(g, i, j))

    return tracker


def _component_probability_generator_igraph(g):
    """
    Provides connected component list and probabilities for each component
    Probability of component i is proportional to n_i^2.

    :param g: igraph.Graph
    :return: list of components, list of probabilities
    """
    # Get list of connected component subgraphs
    components = g.components()
    num_components = len(components)
    component_sizes = [s for s in components.sizes()]

    # Make probabilities proportional to n_i ** 2
    tmp = [s ** 2 for s in component_sizes]
    probabilities = [n / sum(tmp) for n in tmp]
    return components, probabilities
