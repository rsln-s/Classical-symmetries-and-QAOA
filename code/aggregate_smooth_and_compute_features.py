import argparse
import re
import sys
import glob
import pickle
from pathlib import Path
import numpy as np
import networkx as nx
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import progressbar
import copy
import pynauty
from operator import itemgetter
import pandas as pd
from itertools import combinations


def get_adjacency_dict(G):
    """Returns adjacency dictionary for G
    G must be a networkx graph
    Return format: { n : [n1,n2,...], ... }
    where [n1,n2,...] is a list of neighbors of n
    """
    adjacency_dict = {}
    for n, neigh_dict in G.adjacency():
        neigh_list = []
        for neigh, attr_dict in neigh_dict.items():
            assert(len(attr_dict) == 0)
            neigh_list.append(neigh)
        adjacency_dict[n] = neigh_list
    return adjacency_dict


def apply_to_all_ged_k(G, k, f, statistic=np.mean):
    """                              
    f is the function to apply. Must take (G,aut)
    """
    results = []
    edges = [(u,v) for u,v in G.edges()]
    for edges_to_remove in combinations(edges, k):
        G1 = copy.deepcopy(G)
        for e in edges_to_remove:
            assert(len(e) == 2)
            G1.remove_edge(*e)          
        g = pynauty.Graph(number_of_vertices=G1.number_of_nodes(), directed=nx.is_directed(G1),
                adjacency_dict = get_adjacency_dict(G1))
        aut = pynauty.autgrp(g)      
        results.append(f(G1,aut))
    return statistic(results)


def get_shannon_entropy(G, aut):
    S = 0
    for orbit, orbit_size in Counter(aut[3]).items():
        S += ((orbit_size * np.log(orbit_size)) / G.number_of_nodes())
    return S


def get_norbits(G, aut):
    return aut[4]


def get_nauts(G, aut):
    return np.log(aut[1]*(10**aut[2]))


def feature_getter_dispatcher(feature_name, G, aut):
    if feature_name == 'nnodes':
        return G.number_of_nodes()
    elif feature_name == 'shannon_entropy':
        return get_shannon_entropy(G, aut)
    elif feature_name == 'shannon_entropy_ged_1':
        return apply_to_all_ged_k(G, 1, get_shannon_entropy)
    elif feature_name == 'shannon_entropy_ged_2':
        return apply_to_all_ged_k(G, 2, get_shannon_entropy)
    elif feature_name == 'nauts_log':
        return get_nauts(G, aut)
    elif feature_name == 'nauts_log_ged_1':
        return apply_to_all_ged_k(G, 1, get_nauts)
    elif feature_name == 'nauts_log_ged_2':
        return apply_to_all_ged_k(G, 2, get_nauts)
    elif feature_name == 'norbits':
        return get_norbits(G, aut)
    elif feature_name == 'norbits_ged_1':
        return apply_to_all_ged_k(G, 1, get_norbits)
    elif feature_name == 'norbits_ged_2':
        return apply_to_all_ged_k(G, 2, get_norbits)
    else:
        raise ValueError(f"Unknown feature {feature_name}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--files", type = str,
        default = None,
        help = "path to pickles containing the output of min_depth_to_solve_with_smooth_schedule_aposmm.py")
    parser.add_argument(
        "--precomputed", type = Path,
        default = None,
        help = "path to pandas dataframe with previously computed features")
    parser.add_argument(
        "--outpath", type = Path,
        required = True,
        help = "path to dump the result")
    args = parser.parse_args()

    expected_columns = {
        'name',
        'class',
        'elist',
        'p',
        'res'
    }

    expected_features = {
        'nauts_log',
        'nauts_log_ged_1',
        'nauts_log_ged_2',
        'nnodes',
        'norbits',
        'norbits_ged_1',
        'norbits_ged_2',
        'shannon_entropy',
        'shannon_entropy_ged_1',
        'shannon_entropy_ged_2',
    }

    if args.outpath.exists():
        print(f"Found already computed at {args.outpath}, exiting")
        sys.exit()

    if args.precomputed is not None:
        assert(args.precomputed != args.outpath)
        df = pd.read_pickle(args.precomputed) 
        precomputed_names = set(df.index)
        precomputed_columns = set(df.columns) 
        new_columns = expected_features-precomputed_columns
        print(f"New columns (features) to be added: {new_columns}")
        df = df.assign(**{k:np.nan for k in new_columns})
    else:
        df = pd.DataFrame(columns=set.union(expected_columns,expected_features)) 
        df = df.set_index('name')
        precomputed_names = set()

    new_rows = []
    
    print(args.files)
    for fname in progressbar.progressbar(glob.glob(args.files)):
        # TODO: check if the row is already there. If it is, check what fields it already has
        name = Path(fname).name
        if name in precomputed_names:
            # if it's already there, update
            s = df.loc[name]
            existing_columns = set(s.where(s.notna()).dropna().index)
            columns_to_compute = expected_features - existing_columns
            elist = s['elist']
            G = nx.OrderedGraph()
            G.add_edges_from(elist)
            g = pynauty.Graph(number_of_vertices=G.number_of_nodes(), directed=nx.is_directed(G),
                        adjacency_dict = get_adjacency_dict(G))
            aut = pynauty.autgrp(g)
            for feature_name in columns_to_compute:
                df.at[name, feature_name] = float(feature_getter_dispatcher(feature_name, G, aut))
        else:
            # if it's not there yet, build a dictionary with all the stuff
            elist,p,res = pickle.load(open(fname, "rb"))
            d = {'name':name, 'p':float(p), 'elist':copy.deepcopy(elist), 'res':copy.deepcopy(res)}
            G = nx.OrderedGraph()
            G.add_edges_from(elist)
            g = pynauty.Graph(number_of_vertices=G.number_of_nodes(), directed=nx.is_directed(G),
                        adjacency_dict = get_adjacency_dict(G))
            aut = pynauty.autgrp(g)
            for feature_name in expected_features:
                d[feature_name] = float(feature_getter_dispatcher(feature_name, G, aut))
    
            regular_graph_name_regexp = re.compile(r'rand[0-9]reg')
            if 'graph_class' in res:
                d['class'] = res['graph_class']
                # everything below is legacy to support old results
            elif regular_graph_name_regexp.match(d['name']):
                k = G.degree[0]
                assert(G.degree[i] == k for i in range(G.number_of_nodes()))
                d['class'] = f"rand{k}reg"
            elif "trivial" in d['name']:
                d['class'] = 'trivial'
            elif "cycle" in d['name']:
                d['class'] = 'cycle'
            elif "complete" in d['name']:
                d['class'] = 'complete'
            elif "circular_ladder" in d['name']:
                d['class'] = 'circular_ladder'
            elif "ladder" in d['name']:
                d['class'] = 'ladder'
            elif "star_graph" in d['name']:
                d['class'] = 'star_graph'
            else:
                d['class'] = 'hand-picked' 
    
            new_rows.append(copy.deepcopy(d))
    
    if len(new_rows) > 0:
        print(f"Found {len(new_rows)} new points, adding")
        new_df = pd.DataFrame(new_rows, columns=new_rows[0].keys()).set_index('name')
        df = pd.concat([df, new_df], sort=True)
    else:
        print(f"Found nothing new to add!")
    
    print(f"Saving to {args.outpath}")
    df.to_pickle(args.outpath)
