import random
import time
import os
from collections import namedtuple, deque
import numpy as np
import models
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_scatter import scatter_max

random.seed(123)
torch.manual_seed(123)
np.random.seed(123)


class DQAgent:
    ''' deep Q agent '''
    def __init__(self, args):
        '''
        lr: learning rate
        n_step: (s_t-n,a_t-n,r,s_t)
        '''
        self.model_name = args.model
        self.gamma = 0.99 # discount factor of future rewards
        self.n_step = args.n_step # num of steps to accumulate rewards

        self.training = not(args.test)
        self.T = args.T

        self.memory = ReplayMemory(args.memory_size)
        self.batch_size = args.bs # batch size for experience replay

        self.double_dqn = args.double_dqn
        self.device = args.device

        self.node_dim = 2
        self.edge_dim = 4
        self.reg_hidden = args.reg_hidden
        self.embed_dim = args.embed_dim
        # store node embeddings of each graph, avoid multiprocess copy
        self.graph_node_embed = {}
        # model and graph input
        if self.model_name == 'S2V_DUEL':
            self.model = models.S2V_DUEL(reg_hidden=self.reg_hidden, embed_dim=self.embed_dim, node_dim=2, edge_dim=4,
                T=self.T, w_scale=0.01, avg=False).to(self.device)
            # double dqn
            if self.training and self.double_dqn:
                self.target = models.S2V_DUEL(reg_hidden=self.reg_hidden, embed_dim=self.embed_dim, node_dim=2, 
                    edge_dim=4, T=self.T, w_scale=0.01, avg=False).to(self.device)
                self.target.load_state_dict(self.model.state_dict())
                self.target.eval()
            # graph input
            self.setup_graph_input = self.setup_graph_input_s2v

        elif self.model_name == 'S2V_DQN':
            self.model = models.S2V_DQN(reg_hidden=self.reg_hidden, embed_dim=self.embed_dim, node_dim=2, edge_dim=4,
                T=self.T, w_scale=0.01, avg=False).to(self.device)
            # double dqn
            if self.training and self.double_dqn:
                self.target = models.S2V_DQN(reg_hidden=self.reg_hidden, embed_dim=self.embed_dim, node_dim=2, 
                    edge_dim=4, T=self.T, w_scale=0.01, avg=False).to(self.device)
                self.target.load_state_dict(self.model.state_dict())
                self.target.eval()
            # graph input
            self.setup_graph_input = self.setup_graph_input_s2v

        elif self.model_name == 'Tripling':
            self.model = models.Tripling(embed_dim=self.embed_dim, sgate_l1_dim=128, tgate_l1_dim=128, T=3, 
                hidden_dims=[50, 50, 50], w_scale=0.01).to(self.device)
            # double dqn
            if self.training and self.double_dqn:
                self.target = models.Tripling(embed_dim=self.embed_dim, sgate_l1_dim=128, tgate_l1_dim=128, T=3,
                    hidden_dims=[50, 50, 50], w_scale=0.01).to(self.device)
                self.target.load_state_dict(self.model.state_dict())
                self.target.eval()
            # graph input
            self.setup_graph_input = self.setup_graph_input_tripling

        else:
            raise NotImplementedError(f'RL Model {self.model_name}')

        self.criterion = torch.nn.MSELoss(reduction='mean')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)

        if not self.training:
            # load pretrained model for testing
            cwd = os.getcwd()
            self.model.load_state_dict(torch.load(os.path.join(cwd, args.model_file)))
            self.model.eval()

    def reset(self):
        ''' restart '''
        pass

    @torch.no_grad()
    def setup_graph_input_s2v(self, graphs, states, actions=None):
        ''' create a batch data loader from a batch of
                states, # pred all
                states, actions, # pred
                node features (states), edge features
            return a batch from the data loader
        '''
        sample_size = len(graphs)
        data = []
        for i in range(sample_size):
            x = torch.ones(graphs[i].num_nodes, self.node_dim)
            x[:, 1] = 1 - states[i] # selected node feature set 0
            edge_index = torch.tensor(graphs[i].from_to_edges(), dtype=torch.long).t().contiguous()
            edge_attr = torch.ones(graphs[i].num_edges, self.edge_dim)
            edge_attr[:, 1] = torch.tensor([p[-1] for p in graphs[i].from_to_edges_weight()], dtype=torch.float)
            edge_attr[:, 0] = states[i][edge_index[0]]
            edge_attr[:, 2] = torch.abs(states[i][edge_index[0]] - states[i][edge_index[1]])

            y = actions[i].clone() if actions is not None else None

            data.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y))

        loader = DataLoader(data, batch_size=sample_size, shuffle=False)
        for batch in loader:
            # adjust y if applicable
            if actions is not None:
                total_num = 0
                for i in range(1, sample_size):
                    total_num += batch[i - 1].num_nodes
                    batch[i].y += total_num
            return batch.to(self.device)


    def setup_graph_input_tripling(self, graphs, states, actions=None):
        ''' create a batch data loader from a batch of
                states, # pred all
                states, actions, # pred
                node features (states), edge features
            return a batch from the data loader
        '''
        sample_size = len(graphs)
        data = []
        for i in range(sample_size):
            # initialize node embedding if not
            if id(graphs[i]) not in self.graph_node_embed:
                self.graph_node_embed[id(graphs[i])] = models.get_init_node_embed(graphs[i], 30, self.device) # epochs for initial embedding
            with torch.no_grad():
                # copy node embedding as node feature
                x = self.graph_node_embed[id(graphs[i])].detach().clone()
                x = torch.cat((x, states[i].detach().clone().unsqueeze(dim=1)), dim=-1)
                edge_index = torch.tensor(graphs[i].from_to_edges(), dtype=torch.long).t().contiguous()
                # use edge weight 
                edge_weight = torch.tensor([p[-1] for p in graphs[i].from_to_edges_weight()], dtype=torch.float)

                y = actions[i].detach().clone() if actions is not None else None
                data.append(Data(x=x, edge_index=edge_index, edge_weight=edge_weight, y=y))

        with torch.no_grad():
            loader = DataLoader(data, pin_memory=True, num_workers=8, batch_size=sample_size, shuffle=False)
            for batch in loader:
                # adjust y if applicable
                if actions is not None:
                    total_num = 0
                    for i in range(1, sample_size):
                        total_num += batch[i - 1].num_nodes
                        batch[i].y += total_num
                return batch.to(self.device)


    @torch.no_grad()
    def setup_graph_pred(self, graphs, states, actions):
        ''' create a batch data loader from a batch of
                states, actions, 
                node features (states), edge features
            return a batch from the data loader
        '''
        sample_size = len(states)
        data = []
        for i in range(sample_size):
            x = torch.ones(graphs[i].num_nodes, self.node_dim)
            x[:, 1] = 1 - states[i] # selected node feature set 0
            edge_index = torch.tensor(graphs[i].from_to_edges(), dtype=torch.long).t().contiguous()
            edge_attr = torch.ones(graphs[i].num_edges, self.edge_dim)
            edge_attr[:, 1] = torch.tensor([p[-1] for p in graphs[i].from_to_edges_weight()], dtype=torch.float)
            edge_attr[:, 0] = states[i][edge_index[0]]
            edge_attr[:, 2] = torch.abs(states[i][edge_index[0]] - states[i][edge_index[1]])

            y = actions[i].clone()

            data.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y))

        loader = DataLoader(data, batch_size=sample_size, shuffle=False)
        for batch in loader:
            # adjust y
            total_num = torch.tensor([0], dtype=torch.long)
            for i in range(1, sample_size):
                total_num.add_(batch[i - 1].num_nodes)
                batch[i].y = torch.add(batch[i].y, total_num)
            return batch.to(self.device)

    @torch.no_grad()
    def setup_graph_pred_all(self, graph, state):
        ''' create a batch data loader from 
                state,
                node features (state), edge features
            return a batch from the data loader
        '''
        # node features
        x = torch.ones(graph.num_nodes, self.node_dim)
        x[:, 1] = 1 - state # selected node feature set 0
        # from to edges
        edge_index = torch.tensor(graph.from_to_edges(), dtype=torch.long).t().contiguous()
        # edge features
        edge_attr = torch.ones(graph.num_edges, self.edge_dim)
        edge_attr[:, 1] = torch.tensor([p[-1] for p in graph.from_to_edges_weight()], dtype=torch.float)
        edge_attr[:, 0] = state[edge_index[0]]
        edge_attr[:, 2] = torch.abs(state[edge_index[0]] - state[edge_index[1]])
        # creat mini-batch loader
        data = [Data(x=x, edge_index=edge_index, edge_attr=edge_attr)]
        loader = DataLoader(data, batch_size=1, shuffle=False)
        for batch in loader:
            return batch.to(self.device)

    def select_action(self, graph, state, epsilon, training=True, budget=None):
        ''' act upon state '''
        if not(training):
            graph_input = self.setup_graph_input([graph], state.unsqueeze(dim=0))
            with torch.no_grad():
                q_a = self.model(graph_input)
            q_a[state.nonzero()] = -1e5

            if budget is None:
                return torch.argmax(q_a).detach().clone()
            else: # return all seed nodes within budget at one time
                return torch.topk(q_a.squeeze(dim=1), budget)[1].detach().clone()
        # training
        available = (state == 0).nonzero()
        if epsilon > random.random():
            return random.choice(available)
        else:
            graph_input = self.setup_graph_input([graph], state.unsqueeze(dim=0))
            with torch.no_grad():
                q_a = self.model(graph_input)
            max_position = (q_a == q_a[available].max().item()).nonzero()
            return torch.tensor(
                [random.choice(
                    np.intersect1d(available.cpu().contiguous().view(-1).numpy(), 
                        max_position.cpu().contiguous().view(-1).numpy()))], 
                dtype=torch.long)

    def memorize(self, env):
        '''n step for stability'''
        # access state list, reward list and action list from env
        # to add to n step replay memory
        sum_rewards = [0.0]
        for reward in reversed(env.rewards):
            # normalize reward by number of nodes
            reward /= env.graph.num_nodes
            sum_rewards.append(reward + self.gamma * sum_rewards[-1])
        sum_rewards = sum_rewards[::-1]

        for i in range(len(env.states)):
            if i + self.n_step < len(env.states):
                self.memory.push(
                    torch.tensor(env.states[i], dtype=torch.long), 
                    torch.tensor([env.actions[i]], dtype=torch.long), 
                    torch.tensor(env.states[i + self.n_step], dtype=torch.long),
                    torch.tensor([sum_rewards[i] - (self.gamma ** self.n_step) * sum_rewards[i + self.n_step]], dtype=torch.float),
                    env.graph)
            elif i + self.n_step == len(env.states):
                self.memory.push(
                    torch.tensor(env.states[i], dtype=torch.long), 
                    torch.tensor([env.actions[i]], dtype=torch.long), 
                    None,
                    torch.tensor([sum_rewards[i]], dtype=torch.float),  
                    env.graph)


    def fit(self):
        '''fit on a batch sampled from replay memory'''
        # optimize model
        sample_size = self.batch_size if len(self.memory) >= self.batch_size else len(self.memory)
        # need to fix dimension and restrict action space
        transitions = self.memory.sample(sample_size)
        batch = Transition(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), 
            dtype=torch.bool, device=self.device)

        non_final_next_states = [s for s in batch.next_state if s is not None]
        non_final_next_states_graphs = [batch.graph[i] for i, s in enumerate(batch.next_state) if s is not None]

        state_batch = batch.state
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        # graph batch for setting up training batch
        graph_batch = batch.graph

        state_action_values = self.model(self.setup_graph_input(graph_batch, state_batch, action_batch)).squeeze(dim=1)
        next_state_values = torch.zeros(sample_size, device=self.device)

        if len(non_final_next_states) > 0:
            if self.double_dqn:
                batch_non_final = self.setup_graph_input(non_final_next_states_graphs, non_final_next_states)
                next_state_values[non_final_mask] = scatter_max(
                    self.target(batch_non_final).squeeze(dim=1).add_(torch.cat(non_final_next_states).to(self.device) * (-1e5)), 
                    batch_non_final.batch)[0].clamp_(min=0).detach()
            else:
                batch_non_final = self.setup_graph_input(non_final_next_states_graphs, non_final_next_states)
                next_state_values[non_final_mask] = scatter_max(
                    self.model(batch_non_final).squeeze(dim=1).add_(torch.cat(non_final_next_states).to(self.device) * (-1e5)), 
                    batch_non_final.batch)[0].clamp_(min=0).detach()

        expected_state_action_values = next_state_values * self.gamma ** self.n_step + reward_batch.to(self.device)

        loss = self.criterion(state_action_values, expected_state_action_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_net(self):
        # if double dqn, update target network if needed
        if self.double_dqn:
            self.target.load_state_dict(self.model.state_dict())
            return True
        return False

    def save_model(self, file_name):
        cwd = os.getcwd()
        torch.save(self.model.state_dict(), os.path.join(cwd, file_name))


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'graph'))

class ReplayMemory(object):
    '''random replay memory'''
    def __init__(self, capacity):
        # temporily save 1-step snapshot
        self.buffer = deque([], maxlen=capacity)

    def push(self, *args):
        '''Save a transition'''
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


Agent = DQAgent
