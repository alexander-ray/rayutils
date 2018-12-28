import os
import pandas as pd
import numpy as np
"""
Utility function module for small-world analysis
"""


def get_gml_paths(catalog_path, file_root, query):
    """
    Helper function to query existing pandas catalog and retrieve full paths for all GML files in catalog.

    :param catalog_path: Path to existing pandas catalog
    :param file_root: Root for directory containing GML files
    :param query: Query string for pandas
    :return: Tuple containing pandas query results and list of absolute paths
    """
    gmlcatalog = pd.read_pickle(catalog_path)
    simplegmls = gmlcatalog.query(query)
    files = set(simplegmls.index)
    paths = []
    # https://stackoverflow.com/questions/954504/
    for dirpath, dirnames, filenames in os.walk(file_root):
        for filename in [f for f in filenames if f.endswith(".gml")]:
            if filename in files:
                paths.append(os.path.join(dirpath, filename))
    return simplegmls, paths


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
