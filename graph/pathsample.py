import networkx as nx
import numpy as np

def tau_calc_95_rel(mae, n):
    alpha = (0.008145*np.log(n)) - 0.112217
    beta = (-1*0.0451396*np.log(n)) + 0.623809
    ret = (mae - beta)/alpha
    return np.ceil(np.exp(ret)).astype(int)


def tau_calc_95_rel_c_1(mae, n):
    alpha = (0.0038521*np.log(n)) - 0.15962889
    beta = (-1*0.02123643*np.log(n)) + 0.89005047
    ret = (mae - beta)/alpha
    return np.ceil(np.exp(ret)).astype(int)


def tau_calc_95_abs_curve(mae, n):
    num = 1.295 * np.log(n) + 0.481
    return np.ceil((num/mae)**2).astype(int)


def tau_calc_95_rel_curve(mae, n):
    num = (-1 * 0.061) * np.log(n) + 2.114
    return np.ceil((num/mae)**2).astype(int)


def sampler(G, num_samples):
    counter = 0
    for x in range(num_samples):
        i, j = np.random.choice(G.nodes, 2)
        while True:
            try:
                sp = nx.shortest_path_length(G, source=i, target=j, weight=None)
                counter += sp
            except nx.NetworkXNoPath:
                print("chose non allowed path")
                continue
            break
    return counter / num_samples


def sampler_no_rejection(G, num_samples):
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
        sp =  bidirectional_bfs_distance_networkx(subgraph, s=i, t=j)
        tracker += sp
    # Return simple mean
    return tracker / num_samples


def sampler_no_rejection_igraph(G, num_samples):
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
        sp = G.shortest_paths(source=i, target=j, weights=None)[0][0]
        tracker += sp

    # Return simple mean
    return tracker / num_samples


def sampler_no_rejection_igraph_tmp(G, num_samples):
    tracker = 0

    # Get list of connected component subgraphs
    components = G.components()
    num_components = len(components)
    component_sizes = [s for s in components.sizes()]

    # Make probabilities proportional to n_i ** 2
    tmp = [s**2 for s in component_sizes]
    probabilities = [n / sum(tmp) for n in tmp]
    for x in range(num_samples):
        subgraph_index = np.random.choice(num_components, p=probabilities)
        i, j = np.random.choice(components[subgraph_index], 2, replace=True)
        sp = bidirectional_bfs_distance_igraph(G, i, j)
        tracker += sp

    # Return simple mean
    return tracker / num_samples


# Adapted from:
# https://networkx.github.io/documentation/latest/_modules/networkx/algorithms/shortest_paths/unweighted.html#bidirectional_shortest_path
def bidirectional_bfs_distance_networkx(G, s, t):
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


# Adapted from:
# https://networkx.github.io/documentation/latest/_modules/networkx/algorithms/shortest_paths/unweighted.html#bidirectional_shortest_path
def bidirectional_bfs_distance_igraph(G, s, t):
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
