import multiprocessing as mp
import csv
from itertools import product
import numpy as np
import math


# Function must return a list
# Each function will get a single arg from the list of arguments
def multiprocessing_to_csv(fctn, args, filename, num_processes=None, random_initializer=None):
    if not num_processes:
        pool = mp.Pool(initializer=random_initializer)
    else:
        pool = mp.Pool(num_processes, initializer=random_initializer)
    with open(filename, 'a') as f:
        writer = csv.writer(f)
        for result in pool.imap_unordered(fctn, args):
            writer.writerows(result)


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
