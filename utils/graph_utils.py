import copy
import time
import random
import math
import statistics
from collections import deque
import numpy as np
from scipy.sparse import csr_matrix
from multiprocessing import Pool

random.seed(123)
np.random.seed(123)


class Graph:
    ''' graph class '''
    def __init__(self, nodes, edges, children, parents): 
        self.nodes = nodes # set()
        self.edges = edges # dict{(src,dst): weight, }
        self.children = children # dict{node: set(), }
        self.parents = parents # dict{node: set(), }
        # transfer children and parents to dict{node: list, }
        for node in self.children:
            self.children[node] = sorted(self.children[node])
        for node in self.parents:
            self.parents[node] = sorted(self.parents[node])

        self.num_nodes = len(nodes)
        self.num_edges = len(edges)

        self._adj = None
        self._from_to_edges = None
        self._from_to_edges_weight = None

    def get_children(self, node):
        ''' outgoing nodes '''
        return self.children.get(node, [])

    def get_parents(self, node):
        ''' incoming nodes '''
        return self.parents.get(node, [])

    def get_prob(self, edge):
        return self.edges[edge]

    def get_adj(self):
        ''' return scipy sparse matrix '''
        if self._adj is None:
            self._adj = np.zeros((self.num_nodes, self.num_nodes))
            for edge in self.edges:
                self._adj[edge[0], edge[1]] = self.edges[edge] # may contain weight
            self._adj = csr_matrix(self._adj)
        return self._adj

    def from_to_edges(self):
        ''' return a list of edge of (src,dst) '''
        if self._from_to_edges is None:
            self._from_to_edges_weight = list(self.edges.items())
            self._from_to_edges = [p[0] for p in self._from_to_edges_weight]
        return self._from_to_edges

    def from_to_edges_weight(self):
        ''' return a list of edge of (src, dst) with edge weight '''
        if self._from_to_edges_weight is None:
            self.from_to_edges()
        return self._from_to_edges_weight


def read_graph(path, ind=0, directed=False):
    ''' method to load edge as node pair graph '''
    parents = {}
    children = {}
    edges = {}
    nodes = set()

    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not len(line) or line.startswith('#') or line.startswith('%'):
                continue
            row = line.split()
            src = int(row[0]) - ind
            dst = int(row[1]) - ind
            nodes.add(src)
            nodes.add(dst)
            children.setdefault(src, set()).add(dst)
            parents.setdefault(dst, set()).add(src)
            edges[(src, dst)] = 0.0
            if not(directed):
                # regard as undirectional
                children.setdefault(dst, set()).add(src)
                parents.setdefault(src, set()).add(dst)
                edges[(dst, src)] = 0.0

    # change the probability to 1/indegree
    for src, dst in edges:
        edges[(src, dst)] = 1.0 / len(parents[dst])
            
    return Graph(nodes, edges, children, parents)

def computeMC(graph, S, R):
    ''' compute expected influence using MC under IC
        R: number of trials
    '''
    sources = set(S)
    inf = 0
    for _ in range(R):
        source_set = sources.copy()
        queue = deque(source_set)
        while True:
            curr_source_set = set()
            while len(queue) != 0:
                curr_node = queue.popleft()
                curr_source_set.update(child for child in graph.get_children(curr_node) \
                    if not(child in source_set) and random.random() <= graph.edges[(curr_node, child)])
            if len(curr_source_set) == 0:
                break
            queue.extend(curr_source_set)
            source_set |= curr_source_set
        inf += len(source_set)
        
    return inf / R

def workerMC(x):
    ''' for multiprocessing '''
    return computeMC(x[0], x[1], x[2])

def computeRR(graph, S, R, cache=None):
    ''' compute expected influence using RR under IC
        R: number of trials
        The generated RR sets are not saved; 
        We can save those RR sets, then we can use those RR sets
            for any seed set
        cache: maybe already generated list of RR sets for the graph
        l_c: a list of RR set covered, to compute the incremental score
            for environment step
    '''
    # generate RR set
    covered = 0
    generate_RR = False
    if cache is not None:
        if len(cache) > 0:
            # might use break for efficiency for large seed set size or number of RR sets
            return sum(any(s in RR for s in S) for RR in cache) * 1.0 / R * graph.num_nodes
        else:
            generate_RR = True

    for i in range(R):
        # generate one set
        source_set = {random.randint(0, graph.num_nodes - 1)}
        queue = deque(source_set)
        while True:
            curr_source_set = set()
            while len(queue) != 0:
                curr_node = queue.popleft()
                curr_source_set.update(parent for parent in graph.get_parents(curr_node) \
                    if not(parent in source_set) and random.random() <= graph.edges[(parent, curr_node)])
            if len(curr_source_set) == 0:
                break
            queue.extend(curr_source_set)
            source_set |= curr_source_set
        # compute covered(RR) / number(RR)
        for s in S:
            if s in source_set:
                covered += 1
                break
        if generate_RR:
            cache.append(source_set)
    return covered * 1.0 / R * graph.num_nodes


def workerRR(x):
    ''' for multiprocessing '''
    return computeRR(x[0], x[1], x[2])

def computeRR_inc(graph, S, R, cache=None, l_c=None):
    ''' compute expected influence using RR under IC
        R: number of trials
        The generated RR sets are not saved; 
        We can save those RR sets, then we can use those RR sets
            for any seed set
        cache: maybe already generated list of RR sets for the graph
        l_c: a list of RR set covered, to compute the incremental score
            for environment step
    '''
    # generate RR set
    covered = 0
    generate_RR = False
    if cache is not None:
        if len(cache) > 0:
            # might use break for efficiency for large seed set size or number of RR sets
            return sum(any(s in RR for s in S) for RR in cache) * 1.0 / R * graph.num_nodes
        else:
            generate_RR = True

    for i in range(R):
        # generate one set
        source_set = {random.randint(0, graph.num_nodes - 1)}
        queue = deque(source_set)
        while True:
            curr_source_set = set()
            while len(queue) != 0:
                curr_node = queue.popleft()
                curr_source_set.update(parent for parent in graph.get_parents(curr_node) \
                    if not(parent in source_set) and random.random() <= graph.edges[(parent, curr_node)])
            if len(curr_source_set) == 0:
                break
            queue.extend(curr_source_set)
            source_set |= curr_source_set
        # compute covered(RR) / number(RR)
        for s in S:
            if s in source_set:
                covered += 1
                break
        if generate_RR:
            cache.append(source_set)
    return covered * 1.0 / R * graph.num_nodes


if __name__ == '__main__':
    # path of the graph file
    path = "../soc-dolphins.txt"
    # number of parallel processes
    num_process = 5
    # number of trials
    num_trial = 10000
    # load the graph
    graph = read_graph(path, ind=1, directed=False)
    print('Generating seed sets:')
    list_S = []
    for _ in range(10):
      list_S.append(random.sample(range(graph.num_nodes), k=random.randint(3, 10)))
      print(f'({str(list_S[-1])[1:-1]})')

    # cached single-process RR
    print('Cached single-process RR:')
    es_infs = []
    times = []
    time_1 = time.time()
    RR_cache = []
    for S in list_S:
      time_start = time.time()
      es_infs.append(computeRR(graph, S, num_trial, cache=RR_cache))
      times.append(time.time() - time_start)
    time_2 = time.time()

    for i in range(10):
      print(f'({len(list_S[i])}): {list_S[i]}; {times[i]:.2f} seconds; Score {es_infs[i]}')
    print(f'Total gross time: {time_2 - time_1:.2f} seconds')
    print(f'Total time: {sum(times):.2f} seconds')

    # no-cache single-process RR
    print('No-cache single-process RR:')
    es_infs = []
    times = []
    time_1 = time.time()
    for S in list_S:
      time_start = time.time()
      es_infs.append(computeRR(graph, S, num_trial))
      times.append(time.time() - time_start)
    time_2 = time.time()
    for i in range(10):
      print(f'({len(list_S[i])}): {list_S[i]}; {times[i]:.2f} seconds; Score {es_infs[i]}')
    print(f'Total gross time: {time_2 - time_1:.2f} seconds')
    print(f'Total time: {sum(times):.2f} seconds')

    # multi-process MC
    print('Multi-process MC:')
    es_infs = []
    times = []
    time_1 = time.time()
    for S in list_S:
      time_start = time.time()
      with Pool(num_process) as p:
        es_infs.append(statistics.mean(p.map(workerMC, [[graph, S, num_trial // num_process] for _ in range(num_process)])))
      times.append(time.time() - time_start)
    time_2 = time.time()
    for i in range(10):
      print(f'({len(list_S[i])}): {list_S[i]}; {times[i]:.2f} seconds; Score {es_infs[i]}')
    print(f'Total gross time: {time_2 - time_1:.2f} seconds')
    print(f'Total time: {sum(times):.2f} seconds')
