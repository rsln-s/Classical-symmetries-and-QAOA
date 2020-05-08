import sys
import numpy as np
import networkx as nx
import pickle
import argparse
import time
from functools import partial
from pathlib import Path
from variationaltoolkit import VariationalQuantumOptimizerAPOSMM
from variationaltoolkit.utils import brute_force, precompute_obj
from qiskit.optimization.ising.max_cut import get_operator as get_maxcut_operator
from nasa_2020.utils import get_trivial_automorphism_graph_elist

from mpi4py import MPI
is_master = (MPI.COMM_WORLD.Get_rank() == 0)

world_size = MPI.COMM_WORLD.Get_size()

RETURN_FLAG = 42
BLANK_FLAG = 0

def maxcut_obj(x,G):
    cut = 0
    for i, j in G.edges():
        if x[i] != x[j]:
            # the edge is cut
            cut -= 1
    return cut


def find_lowest_p(elist, maxiter, rtol):
    G=nx.OrderedGraph()
    G.add_edges_from(elist)
    
    w = nx.adjacency_matrix(G, nodelist=range(G.number_of_nodes()))
    obj_f = partial(maxcut_obj, G=G)
    C, offset = get_maxcut_operator(w)
    brute_force_opt, _ = brute_force(obj_f, G.number_of_nodes())
    print(f"Brute force optimum: {brute_force_opt}")

    precomputed_energies = precompute_obj(obj_f, G.number_of_nodes())

    init_thetas = [
        [np.pi/4, 0, 0, np.pi/2],
        [np.pi/5, 0, 0, np.pi/2.5],
        [np.pi/6, 0, 0, np.pi/3],
        [np.pi/7, 0, 0, np.pi/3.5],
        [np.pi/8, 0, 0, np.pi/4],
    ]

    for p in range(2, 100):
        start = time.time()
        if is_master:
            start = MPI.Wtime() 
        varopt = VariationalQuantumOptimizerAPOSMM(
            obj_f, 
            'scipy_COBYLA',
            gen_specs_user={'initial_sample_size':len(init_thetas), 'sample_points': init_thetas, 'max_active_runs': len(init_thetas), 'run_max_eval':maxiter},
            variable_bounds = [(0, np.pi/2), (0, np.pi/2), (0, np.pi), (0, np.pi)],
            optimizer_parameters={'maxiter':maxiter}, 
            varform_description={'name':'QAOA', 'p':p, 'cost_operator':C, 'num_qubits':G.number_of_nodes()}, 
            backend_description={'package':'qiskit', 'provider':'Aer', 'name':'statevector_simulator'}, 
            execute_parameters={},
            problem_description={'offset': offset, 'smooth_schedule':True, 'do_not_check_cost_operator':True},
            objective_parameters={'precomputed_energies':precomputed_energies})

        res = varopt.optimize()
        if is_master:
            end = MPI.Wtime()

            print(f"Found {res['min_val']} at p={p} with theta: [{', '.join('{:0.2f}'.format(i) for i in res['opt_params'])}] in {res['num_optimizer_evals']} evals, {end-start:.2f} sec, {G.number_of_nodes()} nodes", flush=True)
            if abs(res['min_val'] - brute_force_opt) < abs(rtol * brute_force_opt):
                for rank in range(1,world_size):
                    MPI.COMM_WORLD.send(RETURN_FLAG, dest=rank) 
                return p, res
            else:
                for rank in range(1,world_size):
                    MPI.COMM_WORLD.send(BLANK_FLAG, dest=rank) 
        else:
            flag = MPI.COMM_WORLD.recv(source=0)
            if flag == RETURN_FLAG:
                return

elist_dict = {
        'trivial_8nodes' : [[0,1], [1,2], [2,3], [3,4], [4,5], [5,6], [4,7]],
        'trivial_9nodes' : [[0,1], [1,2], [2,3], [3,4], [4,5], [5,6], [4,7], [0,8]],
        'k33': [[0,1], [1,2], [2,3], [3,4], [4,5], [5,0], [0,3], [1,4], [2,5]],
        'l3': [[0,1], [1,2], [2,3], [3,4], [4,5], [5,0], [1,4]],
        'k34': [[0,3], [0,4], [0,5], [0,6], [1,3], [1,4], [1,5], [1,6], [2,3], [2,4], [2,5],[2,6]],
        'c5' : [[0,1], [1,2], [2,3], [3,4], [4,0]],
        'c8' : [[0,1], [1,2], [2,3], [3,4], [4,5], [5,6], [6,7], [7,0]],
        'peterson' : [
                [0,1],[1,2],[2,3],[3,4],[4,0],
                [0,5],[1,6],[2,7],[3,8],[4,9],
                [5,7],[5,8],[6,8],[6,9],[7,9]
            ],
        'rand_test' : [(5, 9), (5, 1), (5, 2), (9, 4), (9, 1), (2, 6), (2, 8), (6, 7), (6, 0), (4, 8), (4, 3), (7, 0), (7, 8), (1, 3), (3, 0)],
        'heawood' : [(0, 1), (0, 13), (0, 5), (1, 2), (1, 10), (2, 3), (2, 7), (3, 4), (3, 12), (4, 5), (4, 9), (5, 6), (6, 7), (6, 11), (7, 8), (8, 9), (8, 13), (9, 10), (10, 11), (11, 12), (12, 13)],
        'pappus' : [(0, 1), (0, 17), (0, 5), (1, 2), (1, 8), (2, 3), (2, 13), (3, 4), (3, 10), (4, 5), (4, 15), (5, 6), (6, 7), (6, 11), (7, 8), (7, 14), (8, 9), (9, 10), (9, 16), (10, 11), (11, 12), (12, 13), (12, 17), (13, 14), (14, 15), (15, 16), (16, 17)],
        'desargues' : [(0, 1), (0, 19), (0, 5), (1, 2), (1, 16), (2, 3), (2, 11), (3, 4), (3, 14), (4, 5), (4, 9), (5, 6), (6, 7), (6, 15), (7, 8), (7, 18), (8, 9), (8, 13), (9, 10), (10, 11), (10, 19), (11, 12), (12, 13), (12, 17), (13, 14), (14, 15), (15, 16), (16, 17), (17, 18), (18, 19)],
        'dodecahedral' : [(0, 1), (0, 19), (0, 10), (1, 2), (1, 8), (2, 3), (2, 6), (3, 4), (3, 19), (4, 5), (4, 17), (5, 6), (5, 15), (6, 7), (7, 8), (7, 14), (8, 9), (9, 10), (9, 13), (10, 11), (11, 12), (11, 18), (12, 13), (12, 16), (13, 14), (14, 15), (15, 16), (16, 17), (17, 18), (18, 19)],
        'moebius_kantor' : [(0, 1), (0, 15), (0, 5), (1, 2), (1, 12), (2, 3), (2, 7), (3, 4), (3, 14), (4, 5), (4, 9), (5, 6), (6, 7), (6, 11), (7, 8), (8, 9), (8, 13), (9, 10), (10, 11), (10, 15), (11, 12), (12, 13), (13, 14), (14, 15)],
        'icosahedral' : [(0, 1), (0, 5), (0, 7), (0, 8), (0, 11), (1, 2), (1, 5), (1, 6), (1, 8), (2, 3), (2, 6), (2, 8), (2, 9), (3, 4), (3, 6), (3, 9), (3, 10), (4, 5), (4, 6), (4, 10), (4, 11), (5, 6), (5, 11), (7, 8), (7, 9), (7, 10), (7, 11), (8, 9), (9, 10), (10, 11)]
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gname", type = str,
        default = 'peterson',
        help = f"Name of one of the hardcoded graphs. Must be one of {elist_dict.keys()}")
    parser.add_argument(
        "-n", type = int,
        default = None,
        help = "number of nodes")
    parser.add_argument(
        "-m", type = int,
        default = None,
        help = "width of 2d lattice. args.n is used as the length of the lattice")
    parser.add_argument(
        "-k", type = int,
        default = None,
        help = "degree of each node in randkreg graph")
    parser.add_argument(
        "--periodic",
        action='store_true',
        help = "periodicity of the 2d lattice graph")
    parser.add_argument(
        "--gseed", type = int,
        default = None,
        help = "seed for graph generator")
    parser.add_argument(
        "--maxiter", type = int,
        default = 1000,
        help = "number of iterations, default is 100")
    parser.add_argument(
        "--tol", type = float,
        default = 0.1,
        help = "number of iterations, default is 100")
    args = parser.parse_args()

    graph_class = args.gname
    if 'randkreg' == args.gname:
        # generate random 3 regular graph of size
        assert(args.gseed is not None)
        assert(args.n is not None)
        assert(args.k is not None)
        G = nx.random_regular_graph(args.k, args.n, seed=args.gseed)
        outpath = f"./rand{args.k}reg_n_{args.n}_seed_{args.gseed}_tol_{args.tol}.p"
        elist = [e for e in G.edges]
        del G
    elif 'complete' == args.gname:
        assert(args.n is not None)
        G = nx.generators.classic.complete_graph(args.n)
        outpath = f"./{args.gname}_n_{args.n}_tol_{args.tol}.p"
        elist = [e for e in G.edges]
        del G
    elif 'cycle' == args.gname:
        # assume it's a cycle!
        assert(args.n is not None)
        G = nx.generators.classic.cycle_graph(args.n)
        outpath = f"./{args.gname}_n_{args.n}_tol_{args.tol}.p"
        elist = [e for e in G.edges]
        del G
    elif 'ladder' == args.gname:
        assert(args.n is not None)
        G = nx.generators.classic.ladder_graph(args.n)
        outpath = f"./{args.gname}_n_{args.n}_tol_{args.tol}.p"
        elist = [e for e in G.edges]
        del G
    elif 'circular_ladder' == args.gname:
        assert(args.n is not None)
        G = nx.generators.classic.circular_ladder_graph(args.n)
        outpath = f"./{args.gname}_n_{args.n}_tol_{args.tol}.p"
        elist = [e for e in G.edges]
        del G
    elif 'star_graph' == args.gname:
        assert(args.n is not None)
        G = nx.generators.classic.star_graph(args.n)
        outpath = f"./{args.gname}_n_{args.n}_tol_{args.tol}.p"
        elist = [e for e in G.edges]
        del G
    elif 'wheel' == args.gname:
        assert(args.n is not None)
        G = nx.generators.classic.wheel_graph(args.n)
        outpath = f"./{args.gname}_n_{args.n}_tol_{args.tol}.p"
        elist = [e for e in G.edges]
        del G
    elif 'trivial' == args.gname:
        # generate a graph with a trivial group of automorphisms
        assert(args.n is not None)
        elist = get_trivial_automorphism_graph_elist(args.n) 
        outpath = f"./{args.gname}_{args.n}nodes_tol_{args.tol}.p"
    elif 'antiprism' == args.gname:
        assert(args.n is not None)
        assert(args.n % 2 == 0)
        G = nx.generators.classic.circulant_graph(args.n, [2,1])
        outpath = f"./{args.gname}_{args.n}nodes_tol_{args.tol}.p"
        elist = [e for e in G.edges]
        del G
    elif 'grid_2d' == args.gname:
        assert(args.n is not None)
        assert(args.m is not None)
        G = nx.convert_node_labels_to_integers(nx.generators.lattice.grid_2d_graph(args.m, args.n, periodic=args.periodic))
        outpath = f"./{args.gname}_n_{args.n}_m_{args.n}_period_{args.periodic}_tol_{args.tol}.p"
        elist = [e for e in G.edges]
        del G
    else:
        outpath = f"./{args.gname}_tol_{args.tol}.p"
        elist = elist_dict[args.gname]
        graph_class = 'hand-picked'

    if Path(outpath).exists():
        print(f"Found already computed at {outpath}, exiting")
        sys.exit()
        
    if is_master:
        start_time = MPI.Wtime() 

    t = find_lowest_p(elist, args.maxiter, args.tol)

    if is_master:
        p, res = t
        end_time = MPI.Wtime()
        running_time = end_time-start_time
        print(f"APOSMM finished in {running_time}s with {world_size} processes. Lowest p needed: {p}, saving to {outpath}", flush=True)
        res['graph_class'] = graph_class

        pickle.dump((elist,p,res), open(outpath, "wb"))

    MPI.COMM_WORLD.Barrier()
