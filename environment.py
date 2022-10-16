import numpy as np
import statistics
from multiprocessing import Pool
import random
import time
import utils.graph_utils as graph_utils

random.seed(123)
np.random.seed(123)


class Environment:
    ''' environment that the agents run in '''
    def __init__(self, name, graphs, budget, method='RR', use_cache=False, training=True):
        '''
            method: 'RR' or 'MC'
            use_cache: use cache to speed up
        '''
        # sampled set of graphs
        self.name = name
        self.graphs = graphs
        # IM
        self.budget = budget
        self.method = method
        # useful only if run on the same graph multiple times
        self.use_cache = use_cache
        if self.use_cache:
            if self.method == 'MC':
                # this may be not needed by cached RR
                # not used for RR
                self.influences = {} # cache source set to influence value mapping
            elif self.method == 'RR':
                self.RRs_dict = {}
        self.training = training # whether in training mode or testing mode

    def reset_graphs(self, num_graphs=10):
        # generate new graph
        raise NotImplementedError()

    def reset(self, idx=None, training=True):
        ''' restart '''
        if idx is None:
            self.graph = random.choice(self.graphs)
        else:
            self.graph = self.graphs[idx]
        self.state = [0 for _ in range(self.graph.num_nodes)]
        # IM
        self.prev_inf = 0 # previous influence score
        # store RR sets in case there are more than one graph
        if self.use_cache and self.method == 'RR':
            self.RRs = self.RRs_dict.setdefault(id(self.graph), [])
        self.states = []
        self.actions = []
        self.rewards = []
        self.training = training

    def compute_reward(self, S):
        num_process = 5 # number of parallel processes
        num_trial = 10000 # number of trials
        # fetch influence value
        need_compute = True
        if self.use_cache and self.method == 'MC':
            S_str = f"{id(self.graph)}.{','.join(map(str, sorted(S)))}"
            need_compute = S_str not in self.influences

        if need_compute:
            if self.method == 'MC':
                with Pool(num_process) as p:
                    es_inf = statistics.mean(p.map(graph_utils.workerMC, 
                        [[self.graph, S, int(num_trial / num_process)] for _ in range(num_process)]))
            elif self.method == 'RR':
                if self.use_cache:
                    # cached without incremental
                    es_inf = graph_utils.computeRR(self.graph, S, num_trial, cache=self.RRs)
                else:
                    es_inf = graph_utils.computeRR(self.graph, S, num_trial)
            else:
                raise NotImplementedError(f'{self.method}')

            if self.use_cache and self.method == 'MC':
                self.influences[S_str] = es_inf
        else:
            es_inf = self.influences[S_str]

        reward = es_inf - self.prev_inf
        self.prev_inf = es_inf
        # store reward
        self.rewards.append(reward)

        return reward

    def step(self, node, time_reward=None):
        ''' change state and get reward '''
        # node has already been selected
        if self.state[node] == 1:
            return
        # store state and action
        self.states.append(self.state.copy())
        self.actions.append(node)
        # update state
        self.state[node] = 1
        # calculate reward
        if self.name != 'IM':
            raise NotImplementedError(f'Environment {self.name}')

        S = self.actions
        # whether game is over, budget is reached
        done = len(S) >= self.budget

        if self.training:
            reward = self.compute_reward(S)
        else:
            if done:
                if time_reward is not None:
                    start_time = time.time()
                reward = self.compute_reward(S)
                if time_reward is not None:
                    time_reward[0] = time.time() - start_time
            else:
                reward = None

        return (reward, done)
