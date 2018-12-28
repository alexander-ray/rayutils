import networkx as nx
import numpy as np
import numba as nb
jit = nb.jit


def to_semisparse_matrix(G):
    num_nodes = len(G)

    if num_nodes == 1:
        return 0
    max_degree = max(G.degree, key=lambda x: x[1])[1]
    G = nx.to_scipy_sparse_matrix(G, nodelist=None, dtype=None, weight=None, format='lil')

    def generate_rows(rows):
        for row in rows:
            yield np.pad(row, (0, max_degree - len(row)), 'constant')

    return np.array(np.stack(generate_rows(G.rows)), dtype=np.uint32)


def mean_geodesic_distance(G):
    if nx.is_connected(G):
        return _mean_geodesic_distance_preprocessor(G) / (len(G) * (len(G) - 1))
    else:
        count = 0
        gen = (G.subgraph(c) for c in nx.connected_components(G))
        for g in gen:
            count += _mean_geodesic_distance_preprocessor(g)
        return count / (len(G) * (len(G) - 1))


def _mean_geodesic_distance_preprocessor(G):
    node_list = list(range(len(G)))
    num_nodes = len(G)

    if num_nodes == 1:
        return 0
    max_degree = max(G.degree, key=lambda x: x[1])[1]
    G = nx.to_scipy_sparse_matrix(G, nodelist=None, dtype=None, weight=None, format='lil')

    def generate_rows(rows):
        for row in rows:
            yield np.pad(row, (0, max_degree - len(row)), 'constant')

    tracker = np.full(num_nodes, -1, dtype=np.uint32)
    G_numpy = np.array(np.stack(generate_rows(G.rows)), dtype=np.uint32)
    return _mean_geodesic_distance_driver(G_numpy, num_nodes, node_list, max_degree, tracker)


@jit(nopython=True, nogil=True)
def _mean_geodesic_distance_driver(G, num_nodes, node_list, max_degree, tracker):
    s = 0
    if num_nodes == 1:
        return 0
    for i in node_list:
        s += _bfs_sum_distances(G, i, num_nodes, max_degree, tracker)
    return s


@jit(nopython=True, nogil=True)
def _bfs_sum_distances(G, starting_node, num_nodes, max_degree, tracker):
    pos = 0
    end = 1
    visited = np.zeros(num_nodes)
    dist = np.zeros(num_nodes)

    tracker[pos] = starting_node

    visited[starting_node] = 1
    num_visited = 1

    while not pos > end and num_visited < num_nodes:
        node = tracker[pos]
        pos += 1
        # iterating over neighbors of node
        i = 0
        # Each row is guaranteed to have at least 1 neighbor
        val = G[node, i]
        # If the first element is 0, it therefore a neighbor
        if val == 0:
            if visited[val] == 0:
                visited[val] = 1
                num_visited += 1
                dist[val] = dist[node] + 1
                tracker[end] = val
                end += 1
            i += 1
        # Don't go past end of row
        while i < max_degree and G[node, i] != 0:
            val = G[node, i]
            if visited[val] == 0:
                visited[val] = 1
                num_visited += 1
                dist[val] = dist[node] + 1
                tracker[end] = val
                end += 1
            i += 1
    return np.sum(dist)


def all_pairs_shortest_paths(G):
    if nx.is_connected(G):
        return _all_pairs_shortest_paths_preprocessor(G)
    else:
        results = []
        gen = (G.subgraph(c) for c in nx.connected_components(G))
        for g in gen:
            res = _all_pairs_shortest_paths_preprocessor(g)
            if res is not None:
                results.append(res)
        return np.concatenate(results)


def _all_pairs_shortest_paths_preprocessor(G):
    node_list = list(range(len(G)))
    num_nodes = len(G)

    if num_nodes == 1:
        return np.array([0])
    max_degree = max(G.degree, key=lambda x: x[1])[1]
    G = nx.to_scipy_sparse_matrix(G, nodelist=None, dtype=None, weight=None, format='lil')

    def generate_rows(rows):
        for row in rows:
            yield np.pad(row, (0, max_degree - len(row)), 'constant', constant_values=-1)

    tracker = np.full(num_nodes, -1, dtype=np.int64)
    G_numpy = np.array(np.stack(generate_rows(G.rows)), dtype=np.int64)
    return _all_pairs_shortest_paths_driver(G_numpy, num_nodes, node_list, max_degree, tracker)


@jit(nopython=True, nogil=True)
def _all_pairs_shortest_paths_driver(G, num_nodes, node_list, max_degree, tracker):
    results = np.zeros(num_nodes**2)
    for i in node_list:
        start = i * num_nodes
        results[start:start+num_nodes] = _bfs_distances(G, i, num_nodes, max_degree, tracker)
    return results


@jit(nopython=True, nogil=True)
def _bfs_distances(G, starting_node, num_nodes, max_degree, tracker):
    pos = 0
    end = 1
    visited = np.zeros(num_nodes)
    dist = np.zeros(num_nodes)
    tracker[pos] = starting_node

    visited[starting_node] = 1
    num_visited = 1

    while not pos > end and num_visited < num_nodes:
        node = tracker[pos]
        pos += 1
        # iterating over neighbors of node
        i = 0
        # Don't go past end of row
        while i < max_degree and G[node, i] != -1:
            val = G[node, i]
            if visited[val] == 0:
                visited[val] = 1
                num_visited += 1
                dist[val] = dist[node] + 1
                tracker[end] = val
                end += 1
            i += 1
    return dist
