import networkx as nx
import igraph


def networkx_from_gml(filepath):
    """
    Normal method to read GML file into networkx graph.
    Will convert directed to undirected and converts node labels to integers.

    :param filepath: Path of GML file
    :return: nx.Graph
    """
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

    G.remove_edges_from(G.selfloop_edges())
    return G


def networkx_from_gml_faster(filepath):
    """
    Faster method to read GML file into networkx graph using igraph.
    Reads GML with igraph, converts to undirected, and uses edge list to create corresponding nx.Graph

    :param filepath: Path of GML file
    :return: nx.Graph
    """
    G = igraph.read(filepath)
    G.to_undirected(combine_edges=None)
    G.simplify(multiple=True, loops=True, combine_edges=None)
    edges = [e.tuple for e in G.es]
    G_nx = nx.Graph()
    G_nx.add_edges_from(edges)
    return G_nx


def igraph_from_gml(filepath):
    """
    Helper method to read GML file into igraph.
    Converts to undirected network.

    :param filepath: Path of GML file
    :return: igraph.Graph
    """
    G = igraph.read(filepath)
    G.to_undirected(combine_edges=None)
    G.simplify(multiple=True, loops=True, combine_edges=None)
    return G
