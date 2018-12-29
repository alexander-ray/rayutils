import multiprocessing as mp
import csv


# Function must return a list
# Each function will get a single arg from the list of arguments
def multiprocessing_to_csv(fctn, args, filename, num_processes=None, initializer=None):
    """
    Helper function to parallelize the execution of a function over a set of arguments.

    :param fctn: Function to execute
    :param args: List of arguments. Multiple arguments must be passed as tuple.
    :param filename: Output file
    :param num_processes: Number of processes. Defaults to available number.
    :param initializer: Function to call on initialization of subprocess. Make sure to np.random.seed()!
    """
    if not num_processes:
        pool = mp.Pool(initializer=initializer)
    else:
        pool = mp.Pool(num_processes, initializer=initializer)
    with open(filename, 'a') as f:
        writer = csv.writer(f)
        for result in pool.imap_unordered(fctn, args):
            writer.writerows(result)
