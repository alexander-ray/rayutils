import networkx as nx
import igraph

import pandas as pd
import os


def get_gml_paths(catalog_path, query):
    gmlcatalog = pd.read_pickle(catalog_path)
    simplegmls = gmlcatalog.query(query)
    files = set(simplegmls.index)
    paths = []
    # https://stackoverflow.com/questions/954504/
    for dirpath, dirnames, filenames in os.walk("/Volumes/Samsung_T3/gmls/Anna"):
        for filename in [f for f in filenames if f.endswith(".gml")]:
            if filename in files:
                paths.append(os.path.join(dirpath, filename))
    return paths


def networkx_from_gml(filepath):
    # https://stackoverflow.com/questions/16840554/
    def peek_line(f):
        pos = f.tell()
        line = f.readline()
        f.seek(pos)
        return line

    def skip_gml_comments(f):
        while True:
            line_arr = peek_line(f).strip().split()
            if line_arr and 'graph' == line_arr[0]:
                break
            f.readline()

    with open(filepath, 'r') as f:
        skip_gml_comments(f)
        try:
            G = nx.parse_gml(f)
        except nx.NetworkXError:
            f.seek(0)
            skip_gml_comments(f)
            G = nx.parse_gml(f, label='id')

    G = nx.convert_node_labels_to_integers(G)

    if G.is_directed():
        G = G.to_undirected()

    return G


def networkx_from_gml_faster(filepath):
    G = igraph.read(filepath)
    G.to_undirected(combine_edges='ignore')
    edges = [e.tuple for e in G.es]
    G_nx = nx.Graph()
    G_nx.add_edges_from(edges)
    return G_nx

def igraph_from_gml(filepath):
    G = igraph.read(filepath)
    G.to_undirected(combine_edges='ignore')
    return G
