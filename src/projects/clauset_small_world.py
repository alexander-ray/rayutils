import os
import pandas as pd
from itertools import product
import numpy as np
import math
import csv
"""
Utility function module for small-world analysis
"""


def get_gml_paths(catalog_path, file_root, query, size_limit=None):
    """
    Helper function to query existing pandas catalog and retrieve full paths for all GML files in catalog.

    :param catalog_path: Path to existing pandas catalog
    :param file_root: Root for directory containing GML files
    :param query: Query string for pandas
    :param size_limit: Limit the size of files returned, in bytes
    :return: Tuple containing pandas query results and list of absolute paths
    """
    gmlcatalog = pd.read_pickle(catalog_path)
    simplegmls = gmlcatalog.query(query)
    files = set(simplegmls.index)
    paths = []
    sizes = []
    # https://stackoverflow.com/questions/954504/
    for dirpath, dirnames, filenames in os.walk(file_root):
        for filename in [f for f in filenames if f.endswith(".gml")]:
            if size_limit:
                if filename in files and os.path.getsize(os.path.join(dirpath, filename)) < size_limit:
                    paths.append(os.path.join(dirpath, filename))
                    sizes.append((os.path.join(dirpath, filename), os.path.getsize(os.path.join(dirpath, filename))))
            else:
                if filename in files:
                    paths.append(os.path.join(dirpath, filename))
    return simplegmls, paths, sizes


def get_paths_by_acceptability(catalog_path, file_root):
    acc = set()
    ret = {}
    query = '(Weighted==False) & \
              (Directed==False) & \
              (Bipartite==False) & \
              (Multigraph==False) & \
              (Multiplex==False)'
    _, paths, _ = get_gml_paths(catalog_path, file_root, query)
    ret['ok'] = paths
    acc = acc.union(paths)

    query = '(Weighted==False) & \
                  (Directed==False | Directed==True) & \
                  (Bipartite==False) & \
                  (Multigraph==False) & \
                  (Multiplex==False)'
    _, paths, _ = get_gml_paths(catalog_path, file_root, query)
    ret['ok+directed'] = list(set(paths).difference(acc))
    acc = acc.union(paths)

    query = '(Weighted==False) & \
                      (Directed==False | Directed==True) & \
                      (Bipartite==False) & \
                      (Multigraph==False | Multigraph==True) & \
                      (Multiplex==False)'
    _, paths, _ = get_gml_paths(catalog_path, file_root, query)
    ret['ok+multigraph'] = list(set(paths).difference(acc))
    acc = acc.union(paths)

    query = '(Weighted==False | Weighted==True) & \
              (Directed == False | Directed==True) & \
              (Bipartite == False) & \
              (Multigraph == False | Multigraph==True) & \
              (Multiplex==False | Multiplex==True)'

    _, paths, _ = get_gml_paths(catalog_path, file_root, query)
    ret['iffy'] = list(set(paths).difference(acc))
    acc = acc.union(paths)

    return ret


def remove_paths_by_keywords(paths, keywords):
    assert type(paths) == dict
    for k in paths.keys():
        paths[k] = [e for e in paths[k] if not any((key.lower() in e.lower() for key in keywords))]
    return paths


def cut_down_groups_of_paths(paths, group_keywords, num_to_keep=5):
    assert type(paths) == list
    for kw in group_keywords:
        paths_without = [p for p in paths if kw.lower() not in p.lower()]
        paths_with = [p for p in paths if kw.lower() in p.lower()]
        new_paths_with = [paths_with[i] for i in np.linspace(0, len(paths_with)-1, num_to_keep, dtype=int)]
        paths = paths_without + new_paths_with
    return paths


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


# Lists is list of lists. each argument will be a different combination of each list
# Repetitions is the number of times a combination will be repeated
def assemble_multi_args(lists, repetitions):
    # Make every arg combination
    args = list(product(*lists))
    # If mult samples aren't allowed, for each arg combination repeat as needed
    return [item for item in args for _ in range(repetitions)]


# Specific arg array assembler
# for use in case sampling multiple times from same graph is allowed
def assemble_multi_args_n_tau_multiple_samples_allowed(n_arr, tau_arr, repetitions):
    args = []
    # Iterate through all combinations of args
    for n in n_arr:
        for tau in tau_arr:
            # If n sufficiently large, allow multiple samples per graph
            if n >= 1000:
                # Find how many graphs
                num_different_graphs, samples_per_graph = _get_num_graphs(repetitions, n, tau)
                # May go more than repetitions by maximum (samples_per_graph - 1)
                args.extend([(n, tau, samples_per_graph) for _ in range(num_different_graphs)])
            # Otherwise, allow 1 sample per graph
            else:
                args.extend([(n, tau, 1) for _ in range(repetitions)])
    return args


# For a given number of samples wanted in the end, graph size, num samples per sample
# return number of graphs you need to generate, number of samples per graph
def _get_num_graphs(num_wanted_samples, n, tau):
    samples_from_same_network = _num_samples_allowed_from_same_network(n, tau)
    return int(math.ceil(num_wanted_samples / samples_from_same_network)), samples_from_same_network


# number of times you can sample Tau samples with a 95% probability there will be unique samples
#  using http://preshing.com/20110504/hash-collision-probabilities/
# Polynomial being solved is tau**2 - tau + (ln(0.95) * 2N) where N is number of possibilities
# We let N = n, for smallest number of possible paths
def _num_samples_allowed_from_same_network(n, tau):
    ln_095 = -1 * 0.05129329

    return int(np.max(np.polynomial.polynomial.polyroots([(ln_095 * 2 * n), -1, 1])) // tau) | 1
