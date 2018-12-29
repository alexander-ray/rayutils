import networkx as nx
import numpy as np


def sampler(G, num_samples):
    """
    Rejection sampler for pairwise distances using nx.Graph.
    Too slow for use with disconnected graphs, most of the time.

    :param G: nx.Graph
    :param num_samples: Number of successful samples
    :return: Average pairwise distance
    """
    counter = 0
    for x in range(num_samples):
        i, j = np.random.choice(G.nodes, 2)
        while True:
            try:
                sp = nx.shortest_path_length(G, source=i, target=j, weight=None)
                counter += sp
            except nx.NetworkXNoPath:
                continue
            break
    return counter / num_samples


def sampler_no_rejection(G, num_samples):
    """
    "No-rejection" sampler for pairwise distances using nx.Graph.
    Chooses connected component i with probability proportional to n_i^2.
    Samples pairwise distance within selected component.

    :param G: nx.Graph
    :param num_samples: Number of samples
    :return: Average pairwise distance
    """
    tracker = 0

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
        #sp = nx.shortest_path_length(subgraph, source=i, target=j, weight=None)
        sp = bidirectional_bfs_distance_networkx(subgraph, s=i, t=j)
        tracker += sp
    # Return simple mean
    return tracker / num_samples


def sampler_no_rejection_igraph(G, num_samples):
    """
    igraph version of "no-rejection" sampler for pairwise distances.
    Chooses connected component i with probability proportional to n_i^2.
    Samples pairwise distance within selected component.

    :param G: igraph.Graph
    :param num_samples: Number of samples
    :return: Average pairwise distance
    """
    tracker = 0

    # Get list of connected component subgraphs
    components = G.components()
    num_components = len(components)
    component_sizes = [s for s in components.sizes()]

    # Make probabilities proportional to n_i ** 2
    tmp = [s**2 for s in component_sizes]
    probabilities = [n / sum(tmp) for n in tmp]
    print(probabilities)
    for x in range(num_samples):
        subgraph_index = np.random.choice(num_components, p=probabilities)
        i, j = np.random.choice(components[subgraph_index], 2, replace=True)
        sp = bidirectional_bfs_distance_igraph(G, i, j)
        tracker += sp

    # Return simple mean
    return tracker / num_samples


def bidirectional_bfs_distance_networkx(G, s, t):
    """
    Stripped version of networkx "bidirectional_shortest_path" function.

    :param G: nx.Graph
    :param s: Source node
    :param t: Target node
    :return: Distance between nodes
    """
    if s == t:
        return 0

    # predecesssor and successors in search
    dist_pred = {s: 0}
    dist_succ = {t: 0}

    # initialize fringes, start with forward
    forward_fringe = [s]
    reverse_fringe = [t]

    while forward_fringe and reverse_fringe:
        # Decide which one to work on
        if len(forward_fringe) <= len(reverse_fringe):
            this_level = forward_fringe
            forward_fringe = []
            # Iterate through level
            for v in this_level:
                for w in G.adj[v]:
                    if w not in dist_pred:
                        forward_fringe.append(w)
                        dist_pred[w] = dist_pred[v] + 1
                    if w in dist_succ:  # path found
                        return dist_pred[w] + dist_succ[w]
        else:
            this_level = reverse_fringe
            reverse_fringe = []
            for v in this_level:
                for w in G.adj[v]:
                    if w not in dist_succ:
                        dist_succ[w] = dist_succ[v] + 1
                        reverse_fringe.append(w)
                    if w in dist_pred:  # found path
                        return dist_pred[w] + dist_succ[w]


def bidirectional_bfs_distance_igraph(G, s, t):
    """
    Stripped version of networkx "bidirectional_shortest_path" function, adapted to igraph.

    :param G: igraph.Graph
    :param s: Source node
    :param t: Target node
    :return: Distance between nodes
    """
    if s == t:
        return 0

    # predecesssor and successors in search
    dist_pred = {s: 0}
    dist_succ = {t: 0}

    # initialize fringes, start with forward
    forward_fringe = [s]
    reverse_fringe = [t]

    while forward_fringe and reverse_fringe:
        if len(forward_fringe) <= len(reverse_fringe):
            this_level = forward_fringe
            forward_fringe = []
            for v in this_level:
                for w in G.neighbors(v):
                    if w not in dist_pred:
                        forward_fringe.append(w)
                        dist_pred[w] = dist_pred[v] + 1
                    if w in dist_succ:  # path found
                        return dist_pred[w] + dist_succ[w]
        else:
            this_level = reverse_fringe
            reverse_fringe = []
            for v in this_level:
                for w in G.neighbors(v):
                    if w not in dist_succ:
                        dist_succ[w] = dist_succ[v] + 1
                        reverse_fringe.append(w)
                    if w in dist_pred:  # found path
                        return dist_pred[w] + dist_succ[w]
