import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_add, scatter_softmax, scatter_max
from torch.nn import Embedding
from torch.utils.data import DataLoader
from torch_geometric.utils.num_nodes import maybe_num_nodes
import utils.graph_utils as graph_utils
from collections import deque
from tqdm import tqdm

random.seed(123)
torch.manual_seed(123)
np.random.seed(123)

EPS = 1e-15


class S2V_DQN(nn.Module):
    ''' to check and verify with the design in the paper '''
    def __init__(self, reg_hidden, embed_dim, node_dim, edge_dim, T, w_scale, avg=False):
        '''w_scale=0.01, node_dim=2, edge_dim=4'''
        super(S2V_DQN, self).__init__()
        # depth of structure2vector
        self.T = T # number of graph embedding iterations
        self.embed_dim = embed_dim # p
        self.reg_hidden = reg_hidden
        self.avg = avg

        # input node to latent
        self.w_n2l = torch.nn.Parameter(torch.Tensor(node_dim, embed_dim))
        torch.nn.init.normal_(self.w_n2l, mean=0, std=w_scale)

        # input edge to latent
        self.w_e2l = torch.nn.Parameter(torch.Tensor(edge_dim, embed_dim))
        torch.nn.init.normal_(self.w_e2l, mean=0, std=w_scale)

        # linear node conv
        self.p_node_conv = torch.nn.Parameter(torch.Tensor(embed_dim, embed_dim))
        torch.nn.init.normal_(self.p_node_conv, mean=0, std=w_scale)

        # trans node 1
        self.trans_node_1 = torch.nn.Parameter(torch.Tensor(embed_dim, embed_dim))
        torch.nn.init.normal_(self.trans_node_1, mean=0, std=w_scale)

        # trans node 2
        self.trans_node_2 = torch.nn.Parameter(torch.Tensor(embed_dim, embed_dim))
        torch.nn.init.normal_(self.trans_node_2, mean=0, std=w_scale)

        if self.reg_hidden > 0:
            self.h1_weight = torch.nn.Parameter(torch.Tensor(2 * embed_dim, reg_hidden))
            torch.nn.init.normal_(self.h1_weight, mean=0, std=w_scale)
            self.h2_weight = torch.nn.Parameter(torch.Tensor(reg_hidden, 1))
            torch.nn.init.normal_(self.h2_weight, mean=0, std=w_scale)
            self.last_w = self.h2_weight
        else:
            self.h1_weight = torch.nn.Parameter(torch.Tensor(2 * embed_dim, 1))
            torch.nn.init.normal_(self.h1_weight, mean=0, std=w_scale)
            self.last_w = self.h1_weight

        # S2V scatter message passing
        self.scatter_aggr = (scatter_mean if self.avg else scatter_add)
        

    def forward(self, data):
        '''
           xv: observation, nodes selected are 1, not selected yet are 0
           adj: adjacency matrix of the whole graph

           node_feat, edge_feat, adj in pytorch_geometric batch for varying 
           graph size

           node_input/node_feat: (batch_size x num_node) x node_feat
           edge_input/edge_feat: (batch_size x num_edge) x edge_feat
           adj: (batch_size x num_node) x num_node, sparse might be better
           action_select: batch_size x 1
           data.y: action_select, processed so that it can be directly used 
                   in a batch
           rep_global: in a batch, graph embedding for each node

        '''
        # (batch_size x num_node) x embed_size
        # num_node can vary for different graphs
        data.x = torch.matmul(data.x, self.w_n2l)
        data.x = F.relu(data.x)
        # (batch_size x num_edge) x embed_size
        # num_edge can vary for different graphs
        data.edge_attr = torch.matmul(data.edge_attr, self.w_e2l)

        for _ in range(self.T):
            # (batch_size x num_node) x embed_size
            msg_linear = torch.matmul(data.x, self.p_node_conv)
            # n2esum_param sparse matrix to aggregate node embed to edge embed
            # (batch_size x num_edge) x embed_size
            n2e_linear = msg_linear[data.edge_index[0]]

            # (batch_size x num_edge) x embed_size
            edge_rep = torch.add(n2e_linear, data.edge_attr)
            edge_rep = F.relu(edge_rep)

            # e2nsum_param sparse matrix to aggregate edge embed to node embed
            # (batch_size x num_node) x embed_size
            e2n = self.scatter_aggr(edge_rep, data.edge_index[1], dim=0, dim_size=data.x.size(0))

            # (batch_size x num_node) x embed_size
            data.x = torch.add(torch.matmul(e2n, self.trans_node_1), 
                               torch.matmul(data.x, self.trans_node_2))
            data.x = F.relu(data.x)

        # subgsum_param sparse matrix to aggregate node embed to graph embed
        # batch_size x embed_size
        # torch_scatter can do broadcasting
        y_potential = self.scatter_aggr(data.x, data.batch, dim=0)
        # can concatenate budget to global representation
        if data.y is not None: # Q func given a
            # batch_size x embed_size
            action_embed = data.x[data.y]

            # batch_size x (2 x embed_size)
            embed_s_a = torch.cat((action_embed, y_potential), dim=-1) # ConcatCols

            last_output = embed_s_a
            if self.reg_hidden > 0:
                # batch_size x reg_hidden
                hidden = torch.matmul(embed_s_a, self.h1_weight)
                last_output = F.relu(hidden)
            # batch_size x 1
            q_pred = torch.matmul(last_output, self.last_w)

            return q_pred

        else: # Q func on all a
            rep_y = y_potential[data.batch]
            embed_s_a_all = torch.cat((data.x, rep_y), dim=-1) # ConcatCols

            last_output = embed_s_a_all
            if self.reg_hidden > 0:
                hidden = torch.matmul(embed_s_a_all, self.h1_weight)
                last_output = torch.relu(hidden)

            q_on_all = torch.matmul(last_output, self.last_w)

            return q_on_all


class Tripling(nn.Module):
    ''' the tripling GNN RL model '''
    def __init__(self, embed_dim, sgate_l1_dim, tgate_l1_dim, T, hidden_dims, w_scale):
        '''
        embed_dim=50 -> 2*50+1
        T: number of layers of tripling
        '''
        super(Tripling, self).__init__()
        self.embed_dim = embed_dim
        self.sgate_l1_dim = sgate_l1_dim
        self.tgate_l1_dim = tgate_l1_dim
        self.T = T
        self.hidden_dims = hidden_dims.copy() 
        # also include initial embed dim
        self.hidden_dims.insert(0, embed_dim)
        self.trans_weights = nn.ParameterList()
        # state GNN
        self.influgate_etas = nn.ParameterList()
        self.state_weights_self = nn.ParameterList()
        self.state_weights_neibor = nn.ParameterList()
        # weights for edge attention and edge weight
        self.state_weights_attention = nn.ParameterList()
        self.state_weights_edge = nn.ParameterList()

        # source GNN
        self.source_betas = nn.ParameterList()
        self.sourcegate_layer1s = nn.ModuleList()
        self.sourcegate_layer2s = nn.ModuleList()
        self.source_weights_self = nn.ParameterList()
        self.source_weights_neibor = nn.ParameterList()
        self.source_weights_state = nn.ParameterList()
        # weights for edge attention and edge weight
        self.source_weights_attention = nn.ParameterList()
        self.source_weights_edge = nn.ParameterList()

        # target GNN
        self.target_taus = nn.ParameterList()
        self.targetgate_layer1s = nn.ModuleList()
        self.targetgate_layer2s = nn.ModuleList()
        self.target_weights_self = nn.ParameterList()
        self.target_weights_neibor = nn.ParameterList()
        self.target_weights_state = nn.ParameterList()
        # weights for edge attention and edge weight
        self.target_weights_attention = nn.ParameterList()
        self.target_weights_edge = nn.ParameterList()

        for i in range(1, T+1):
            self.trans_weights.append(torch.nn.Parameter(torch.Tensor(self.hidden_dims[i-1], self.hidden_dims[i])))
            torch.nn.init.normal_(self.trans_weights[-1], mean=0, std=w_scale)
            ######## state GNN ########
            self.influgate_etas.append(torch.nn.Parameter(torch.Tensor(2*self.hidden_dims[i], 1)))
            torch.nn.init.normal_(self.influgate_etas[-1], mean=0, std=w_scale)
            self.state_weights_self.append(torch.nn.Parameter(torch.Tensor(1)))
            torch.nn.init.normal_(self.state_weights_self[-1], mean=0, std=w_scale)
            self.state_weights_neibor.append(torch.nn.Parameter(torch.Tensor(1)))
            torch.nn.init.normal_(self.state_weights_neibor[-1], mean=0, std=w_scale)
            self.state_weights_attention.append(torch.nn.Parameter(torch.Tensor(1)))
            torch.nn.init.normal_(self.state_weights_attention[-1], mean=0, std=w_scale)
            self.state_weights_edge.append(torch.nn.Parameter(torch.Tensor(1)))
            torch.nn.init.normal_(self.state_weights_edge[-1], mean=0, std=w_scale)

            ######## source GNN ########
            self.source_betas.append(torch.nn.Parameter(torch.Tensor(2*self.hidden_dims[i], 1)))
            torch.nn.init.normal_(self.source_betas[-1], mean=0, std=w_scale)
            self.sourcegate_layer1s.append(torch.nn.Linear(self.hidden_dims[i-1], sgate_l1_dim, True))
            self.sourcegate_layer2s.append(torch.nn.Linear(sgate_l1_dim, self.hidden_dims[i], True))
            self.source_weights_self.append(torch.nn.Parameter(torch.Tensor(1)))
            torch.nn.init.normal_(self.source_weights_self[-1], mean=0, std=w_scale)
            self.source_weights_neibor.append(torch.nn.Parameter(torch.Tensor(1)))
            torch.nn.init.normal_(self.source_weights_neibor[-1], mean=0, std=w_scale)
            self.source_weights_state.append(torch.nn.Parameter(torch.Tensor(1)))
            torch.nn.init.normal_(self.source_weights_state[-1], mean=0, std=w_scale)
            self.source_weights_attention.append(torch.nn.Parameter(torch.Tensor(1)))
            torch.nn.init.normal_(self.source_weights_attention[-1], mean=0, std=w_scale)
            self.source_weights_edge.append(torch.nn.Parameter(torch.Tensor(1)))
            torch.nn.init.normal_(self.source_weights_edge[-1], mean=0, std=w_scale)

            ######## target GNN ########
            self.target_taus.append(torch.nn.Parameter(torch.Tensor(2*self.hidden_dims[i], 1)))
            torch.nn.init.normal_(self.target_taus[-1], mean=0, std=w_scale)
            self.targetgate_layer1s.append(torch.nn.Linear(self.hidden_dims[i-1], tgate_l1_dim, True))
            self.targetgate_layer2s.append(torch.nn.Linear(tgate_l1_dim, self.hidden_dims[i], True))
            self.target_weights_self.append(torch.nn.Parameter(torch.Tensor(1)))
            torch.nn.init.normal_(self.target_weights_self[-1], mean=0, std=w_scale)
            self.target_weights_neibor.append(torch.nn.Parameter(torch.Tensor(1)))
            torch.nn.init.normal_(self.target_weights_neibor[-1], mean=0, std=w_scale)
            self.target_weights_state.append(torch.nn.Parameter(torch.Tensor(1)))
            torch.nn.init.normal_(self.target_weights_state[-1], mean=0, std=w_scale)
            self.target_weights_attention.append(torch.nn.Parameter(torch.Tensor(1)))
            torch.nn.init.normal_(self.target_weights_attention[-1], mean=0, std=w_scale)
            self.target_weights_edge.append(torch.nn.Parameter(torch.Tensor(1)))
            torch.nn.init.normal_(self.target_weights_edge[-1], mean=0, std=w_scale)

        # dqn
        self.theta1 = torch.nn.Parameter(torch.Tensor(3*self.hidden_dims[-1], 1))
        torch.nn.init.normal_(self.theta1, mean=0, std=w_scale)
        self.theta2 = torch.nn.Parameter(torch.Tensor(self.hidden_dims[-1], self.hidden_dims[-1]))
        torch.nn.init.normal_(self.theta2, mean=0, std=w_scale)
        self.theta3 = torch.nn.Parameter(torch.Tensor(self.hidden_dims[-1], self.hidden_dims[-1]))
        torch.nn.init.normal_(self.theta3, mean=0, std=w_scale)
        self.theta4 = torch.nn.Parameter(torch.Tensor(self.hidden_dims[-1], self.hidden_dims[-1]))
        torch.nn.init.normal_(self.theta4, mean=0, std=w_scale)


    def forward(self, data):
        ''' take initial node embedding as input,
            do tripling message propogation
        '''
        source_influ = data.x[:, :self.hidden_dims[0]]
        target_influ = data.x[:, self.hidden_dims[0]:2*self.hidden_dims[0]]
        state = data.x[:, -1]
        for i in range(0, self.T):
            trans_source_influ = torch.matmul(source_influ, self.trans_weights[i])
            trans_target_influ = torch.matmul(target_influ, self.trans_weights[i])
            trans_influ = torch.cat((trans_source_influ[data.edge_index[0]], trans_target_influ[data.edge_index[1]]), dim=-1)
            ######## state GNN ########
            e_uv = torch.matmul(trans_influ, self.influgate_etas[i]).squeeze(dim=1)
            e_uv = F.leaky_relu(e_uv, 0.2)
            influgate = scatter_softmax(e_uv, data.edge_index[1])
            # weighted sum of attention and edge weight
            influgate = influgate * self.state_weights_attention[i] + data.edge_weight * self.state_weights_edge[i]

            a_v = scatter_add(influgate * state[data.edge_index[0]], data.edge_index[1], dim_size=data.x.size(0))
            new_state = torch.sigmoid(state * self.state_weights_self[i] + a_v * self.state_weights_neibor[i])
            new_state = new_state * (1 - data.x[:, -1]) + data.x[:, -1]

            ######## source GNN ########
            f_vw = torch.matmul(trans_influ, self.source_betas[i]).squeeze(dim=1)
            f_vw = F.leaky_relu(f_vw, 0.2)
            alpha_vw = scatter_softmax(f_vw, data.edge_index[0])
            # weighted sum of attention and edge weight
            alpha_vw = alpha_vw * self.source_weights_attention[i] + data.edge_weight * self.source_weights_edge[i]

            sourcegate = F.leaky_relu(self.sourcegate_layer2s[i](F.leaky_relu(self.sourcegate_layer1s[i](target_influ[data.edge_index[1]]), 0.2)), 0.2)
            b_v = scatter_add(alpha_vw.unsqueeze(dim=1) * sourcegate, data.edge_index[0], dim=0, dim_size=data.x.size(0))
            new_source_influ = F.leaky_relu(trans_source_influ * self.source_weights_self[i] + b_v * self.source_weights_neibor[i] + (state * self.source_weights_state[i]).unsqueeze(dim=1))
            
            ######## target GNN ########
            d_uv = torch.matmul(trans_influ, self.target_taus[i]).squeeze(dim=1)
            d_uv = F.leaky_relu(d_uv, 0.2)
            phi_uv = scatter_softmax(d_uv, data.edge_index[1])
            # weighted sum of attention and edge weight
            phi_uv = phi_uv * self.target_weights_attention[i] + data.edge_weight * self.target_weights_edge[i]

            targetgate = F.leaky_relu(self.targetgate_layer2s[i](F.leaky_relu(self.targetgate_layer1s[i](source_influ[data.edge_index[0]]), 0.2)), 0.2)
            c_v = scatter_add(phi_uv.unsqueeze(dim=1) * targetgate, data.edge_index[1], dim=0, dim_size=data.x.size(0))
            new_target_influ = F.leaky_relu(trans_target_influ * self.target_weights_self[i] + c_v * self.target_weights_neibor[i] + (state * self.target_weights_state[i]).unsqueeze(dim=1))

            state = new_state
            source_influ = new_source_influ
            target_influ = new_target_influ

        # RL (threshold or minus X; r-hop out-neibors or all nodes)
        # minus X, all nodes
        if data.y is not None: # Q func given a
            # calculate the S_a to all T not activated
            S_v = source_influ[data.y] # only action nodes

            # not the action node
            not_y = torch.ones(target_influ.size(0), dtype=torch.bool, device=data.y.device)
            not_y[data.y] = False
            # not active nodes
            not_selected = data.x[:, -1] == 0
            not_idx = torch.logical_and(not_y, not_selected)

            batch_idx = data.batch[not_idx]
            T_u = target_influ[not_idx]

            # get seed nodes
            is_idx = data.x[:, -1] == 1
            batch_is_idx = data.batch[is_idx]
            S_w = source_influ[is_idx]

            q_pred = torch.matmul(F.leaky_relu(torch.cat([torch.matmul(S_v, self.theta2), torch.matmul(scatter_add(S_w, batch_is_idx, dim=0, dim_size=data.batch[-1].item()+1), self.theta4), torch.matmul(scatter_add(T_u, batch_idx, dim=0, dim_size=data.batch[-1].item()+1), self.theta3)], dim=-1)), self.theta1)

            return q_pred
        else: # Q func on all a
            target_influ[data.x[:, -1] == 1] = 0.0 # not count the activation to already active nodes
            state[data.x[:, -1] == 1] = 1.0 # set activate node state to 1.0

            source_influ_copy = source_influ.clone()
            # sum up only seed nodes for each mini-graph
            source_influ[data.x[:, -1] == 0] = 0.0
            source_influ_w = scatter_add(source_influ, data.batch, dim=0).repeat_interleave(scatter_add(torch.ones(data.batch.size(0), dtype=torch.long, device=data.batch.device), data.batch), dim=0)
            # no need to calculate Q value for a seed as action
            source_influ_w[data.x[:, -1] == 1] = 0.0
            source_influ_copy[data.x[:, -1] == 1] = 0.0
            
            q_on_all = torch.matmul(F.leaky_relu(torch.cat([torch.matmul(source_influ_copy, self.theta2), torch.matmul(source_influ_w, self.theta4), torch.matmul(scatter_add(target_influ, data.batch, dim=0).repeat_interleave(scatter_add(torch.ones(data.batch.size(0), dtype=torch.long, device=data.batch.device), data.batch), dim=0) - target_influ, self.theta3)], dim=-1)), self.theta1)

            return q_on_all


def get_init_node_embed(graph, num_epochs, device):
    ''' use deep walk to generate initial node embedding 
        according to the graph structure (node connection and edge weight)
    '''
    model = DeepWalkNeg(graph, embedding_dim=50, walk_length=3, r_hop=5, r_hop_size=5, 
        walks_per_node=50, num_negative_samples=5, restart=0.15, sparse=True).to(device)

    loader = model.loader(batch_size=32, shuffle=True, num_workers=4)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

    def train():
        model.train()
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    for epoch in range(num_epochs):
        loss = train()

    return model().detach().cpu().clone()


class DeepWalkNeg(nn.Module):

    def __init__(self, graph, embedding_dim, walk_length, r_hop, r_hop_size,
                 walks_per_node=1, num_negative_samples=1, restart=0.5, sparse=False):
        super().__init__()

        self.graph = graph
        self.embedding_dim = embedding_dim # dimension of S / T, dim(X)=1
        self.walk_length = walk_length - 1 # number of steps of walking
        self.walks_per_node = walks_per_node
        self.r_hop = r_hop
        self.r_hop_size = r_hop_size
        self.restart = restart
        self.num_negative_samples = num_negative_samples

        self.embedding = Embedding(graph.num_nodes, embedding_dim*2+1, sparse=sparse)

        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()

    def forward(self, batch=None):
        ''' return the embeddings for the nodes in :obj:`batch`. '''
        emb = self.embedding.weight
        return emb if batch is None else emb.index_select(0, batch)

    def loader(self, **kwargs):
        return DataLoader(range(self.graph.num_nodes),
                          collate_fn=self.sample, **kwargs)

    def random_walk(self, start, walk_len, restart=0.5, rand=random.Random()):
        ''' generate random walks following deep walk 
        Args:
            start: start node
            walk_length: the length of one walk
            restart: restart probability
            rand: used to randomly select the next node in a walk
        '''
        path = [start]
        for _ in range(walk_len):
            cur = path[-1]
            if len(self.graph.get_children(cur)) > 0 and rand.random() >= restart:
                path.append(rand.choice(self.graph.get_children(cur)))
            else: # also restart if can't walk forward
                path.append(path[0])
        return path

    def r_hop_neibors(self, start, r_hop):
        ''' return r hop neibors by breadth-first search '''
        nodes_set = {start}
        queue = deque(nodes_set)
        for _ in range(r_hop):
            curr_nodes = set()
            while len(queue) != 0:
                curr_node = queue.popleft()
                curr_nodes.update(child for child in self.graph.get_children(curr_node) if not(child in nodes_set))
            if len(curr_nodes) == 0:
                break
            queue.extend(curr_nodes)
            nodes_set |= curr_nodes
        return list(nodes_set)

    def pos_sample(self, batch):
        ''' can rewrite to parallelism '''
        batch = batch.repeat(self.walks_per_node)
        r_hop_n = {}
        walks = []
        for b in batch:
            s = b.item()
            rw = self.random_walk(s, self.walk_length, restart=self.restart)
            # sampling from r-hop context
            if s not in r_hop_n:
                r_hop_n[s] = self.r_hop_neibors(s, self.r_hop)
            rw.extend(random.choices(r_hop_n[s], k=self.r_hop_size))

            walks.append(rw)
        return torch.tensor(walks, dtype=torch.long)

    def neg_sample(self, batch):
        batch = batch.repeat(self.walks_per_node * self.num_negative_samples)

        rw = torch.randint(self.graph.num_nodes,
                           (batch.size(0), self.walk_length+self.r_hop_size))
        rw = torch.cat([batch.view(-1, 1), rw], dim=-1)

        return rw

    def sample(self, batch):
        if not isinstance(batch, torch.Tensor):
            batch = torch.tensor(batch)
        return self.pos_sample(batch), self.neg_sample(batch)

    def loss(self, pos_rw, neg_rw):
        ''' computes the loss given positive and negative random walks. '''

        # Positive loss
        start, rest = pos_rw[:, 0], pos_rw[:, 1:].contiguous()

        h_start = self.embedding(start).view(pos_rw.size(0), 1, self.embedding_dim*2+1)
        h_rest = self.embedding(rest.view(-1)).view(pos_rw.size(0), -1, self.embedding_dim*2+1)

        # start: u, context: v
        # Xu * Su * Tv + Xv
        out = (h_start[:, :, -1] * (h_start[:, :, :self.embedding_dim] * h_rest[:, :, self.embedding_dim:self.embedding_dim*2]).sum(dim=-1) + h_rest[:, :, -1]).view(-1)

        pos_loss = -torch.log(torch.sigmoid(out) + EPS).mean()

        # Negative loss
        start, rest = neg_rw[:, 0], neg_rw[:, 1:].contiguous()

        h_start = self.embedding(start).view(neg_rw.size(0), 1, self.embedding_dim*2+1)
        h_rest = self.embedding(rest.view(-1)).view(neg_rw.size(0), -1, self.embedding_dim*2+1)

        # start: u, context: v
        # Xu * Su * Tv + Xv
        out = (h_start[:, :, -1] * (h_start[:, :, :self.embedding_dim] * h_rest[:, :, self.embedding_dim:self.embedding_dim*2]).sum(dim=-1) + h_rest[:, :, -1]).view(-1)

        neg_loss = -torch.log(1 - torch.sigmoid(out) + EPS).mean()

        return pos_loss + neg_loss

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.embedding.weight.size(0)}, '
                f'{self.embedding.weight.size(1)})')


class S2V_DUEL(nn.Module):
    ''' structure2vector + dueling deep Q network '''
    def __init__(self, reg_hidden, embed_dim, len_pre_pooling, len_post_pooling, T):
        super(S2V_DUEL, self).__init__()
        # depth of structure2vector
        self.T = T
        self.embed_dim = embed_dim
        self.reg_hidden = reg_hidden
        self.len_pre_pooling = len_pre_pooling
        self.len_post_pooling = len_post_pooling
        self.mu_1 = torch.nn.Parameter(torch.Tensor(1, embed_dim))
        torch.nn.init.normal_(self.mu_1, mean=0, std=0.01)
        self.mu_2 = torch.nn.Linear(embed_dim, embed_dim, True)
        torch.nn.init.normal_(self.mu_2.weight, mean=0, std=0.01)

        self.list_pre_pooling = []
        for i in range(self.len_pre_pooling):
            pre_lin = torch.nn.Linear(embed_dim, embed_dim, bias=True)
            torch.nn.init.normal_(pre_lin.weight, mean=0, std=0.01)
            self.list_pre_pooling.append(pre_lin)

        self.list_post_pooling = []
        for i in range(self.len_post_pooling):
            post_lin =torch.nn.Linear(embed_dim, embed_dim, bias=True)
            torch.nn.init.normal_(post_lin.weight, mean=0, std=0.01)
            self.list_post_pooling.append(post_lin)
        self.q_1 = torch.nn.Linear(embed_dim, embed_dim, bias=True)
        torch.nn.init.normal_(self.q_1.weight, mean=0, std=0.01)
        self.q_2 = torch.nn.Linear(embed_dim, embed_dim, bias=True)
        torch.nn.init.normal_(self.q_2.weight, mean=0, std=0.01)

        if self.reg_hidden > 0:
            self.q_reg = torch.nn.Linear(2 * embed_dim, self.reg_hidden)
            torch.nn.init.normal_(self.q_reg.weight, mean=0, std=0.01)
            # define dueling
            self.fc_value = torch.nn.Linear(self.reg_hidden, 4 * self.reg_hidden)
            self.fc_adv = torch.nn.Linear(self.reg_hidden, 4 * self.reg_hidden)
            self.value = torch.nn.Linear(4 * self.reg_hidden, 1)
            self.adv = torch.nn.Linear(4 * self.reg_hidden, 1)
        else:
            # define dueling
            self.fc_value = torch.nn.Linear(2 * embed_dim, 8 * embed_dim)
            self.fc_adv = torch.nn.Linear(2 * embed_dim, 8 * embed_dim)
            self.value = torch.nn.Linear(8 * embed_dim, 1)
            self.adv = torch.nn.Linear(8 * embed_dim, 1)

        torch.nn.init.normal_(self.fc_value.weight, mean=0, std=0.01)
        torch.nn.init.normal_(self.fc_adv.weight, mean=0, std=0.01)
        torch.nn.init.normal_(self.value.weight, mean=0, std=0.01)
        torch.nn.init.normal_(self.adv.weight, mean=0, std=0.01)

    def forward(self, xv, adj):
        # structure2vector
        minibatch_size = xv.shape[0]
        num_node = xv.shape[1]
        for t in range(self.T):
            if t == 0:
                mu = torch.matmul(xv, self.mu_1).clamp(0) # used as ReLU
            else:
                mu_1 = torch.matmul(xv, self.mu_1).clamp(0)
                # before pooling:
                for i in range(self.len_pre_pooling):
                    mu = self.list_pre_pooling[i](mu).clamp(0)
                mu_pool = torch.matmul(adj, mu)
                # after pooling
                for i in range(self.len_post_pooling):
                    mu_pool = self.list_post_pooling[i](mu_pool).clamp(0)
                mu_2 = self.mu_2(mu_pool)
                mu = torch.add(mu_1, mu_2).clamp(0)
        # Q network
        q_1 = self.q_1(torch.matmul(xv.transpose(1,2), mu)).expand(minibatch_size, num_node, self.embed_dim)
        q_2 = self.q_2(mu)
        q_ = torch.cat((q_1, q_2), dim=-1)
        if self.reg_hidden > 0:
            q_reg = self.q_reg(q_).clamp(0)
            # insert dueling 
            value = self.fc_value(torch.mean(q_reg, dim=1, keepdim=True)).clamp(0) # aggregate over all nodes embeddings
            adv = self.fc_adv(q_reg).clamp(0)

            value = self.value(value)
            adv = self.adv(adv)

            advAverage = torch.mean(adv, dim=1, keepdim=True) # aggregate advantage over all nodes
            q = value + adv - advAverage
        else:
            q_= q_.clamp(0)
            # insert dueling
            value = self.fc_value(torch.mean(q_, dim=1, keepdim=True)).clamp(0)
            adv = self.fc_adv(q_).clamp(0)

            value = self.value(value)
            adv = self.adv(adv)

            advAverage = torch.mean(adv, dim=1, keepdim=True)
            q = value + adv - advAverage
        return q



