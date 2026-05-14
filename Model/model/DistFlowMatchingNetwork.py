from typing import Optional, Tuple
import torch
from torch import nn
from torch_geometric.nn import MessagePassing
import torch_geometric
from torch.autograd import grad
import torch_geometric.utils
from torch_scatter import scatter, scatter_add
import math
import torch.nn.functional as F
from copy import deepcopy

import math

from torchdiffeq import odeint_adjoint

class NodeEncoding(nn.Module):
    def __init__(self, 
                 out_dim, 
                 cutoff_lower, 
                 cutoff_upper, 
                 num_rbf, 
                 max_z,
                 time_embedding=True):
        super(NodeEncoding, self).__init__()
        self.hidden_channels = out_dim
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.num_rbf = num_rbf
        self.max_z = max_z
        self.embedding = nn.Embedding(self.max_z, self.hidden_channels)
        
        self.time_embedding = TimestepEmbedding(self.hidden_channels) if time_embedding else None
        self.neighbor_embedding = NeighborEmbedding(self.hidden_channels, self.num_rbf, self.cutoff_lower, self.cutoff_upper)

    def reset_parameters(self):
        self.embedding.reset_parameters()
        self.neighbor_embedding.reset_parameters()

    def forward(self, z, edge_index, edge_dist, rbf_attr, t):
        x = self.embedding(z)

        if self.time_embedding is not None:
            x = x + self.time_embedding(t)

        if self.neighbor_embedding is not None:
            x = self.neighbor_embedding(z, x, edge_index, edge_dist, rbf_attr)

        return x

class MultiHeadAttention(MessagePassing):
    def __init__(
        self,
        hidden_channels,
        num_rbf,
        distance_influence,
        num_heads,
        activation,
        attn_activation,
        cutoff_lower,
        cutoff_upper,
    ):
        super(MultiHeadAttention, self).__init__(aggr="add", node_dim=0)
        assert hidden_channels % num_heads == 0, (
            f"The number of hidden channels ({hidden_channels}) "
            f"must be evenly divisible by the number of "
            f"attention heads ({num_heads})"
        )

        self.distance_influence = distance_influence
        self.num_heads = num_heads
        self.hidden_channels = hidden_channels
        self.head_dim = hidden_channels // num_heads

        self.act = act_class_mapping[activation]()
        self.attn_activation = act_class_mapping[attn_activation]()
        self.cutoff = CosineCutoff(cutoff_lower, cutoff_upper)

        self.q_proj = nn.Linear(hidden_channels, hidden_channels)
        self.k_proj = nn.Linear(hidden_channels, hidden_channels)
        self.v_proj = nn.Linear(hidden_channels, hidden_channels)
        self.o_proj = nn.Linear(hidden_channels, hidden_channels)

        self.edge_proj = nn.Linear(hidden_channels, hidden_channels)

        self.dk_proj = None
        if distance_influence in ["keys", "both"]:
            self.dk_proj = nn.Linear(num_rbf, hidden_channels)

        self.dv_proj = None
        if distance_influence in ["values", "both"]:
            self.dv_proj = nn.Linear(num_rbf, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        self.q_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.k_proj.weight)
        self.k_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.v_proj.weight)
        self.v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.edge_proj.weight)
        self.edge_proj.bias.data.fill_(0)
        if self.dk_proj:
            nn.init.xavier_uniform_(self.dk_proj.weight)
            self.dk_proj.bias.data.fill_(0)
        if self.dv_proj:
            nn.init.xavier_uniform_(self.dv_proj.weight)
            self.dv_proj.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_attr, r_ij, f_ij):
        q = self.q_proj(x).reshape(-1, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(-1, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(-1, self.num_heads, self.head_dim)

        e = self.edge_proj(edge_attr).reshape(-1, self.num_heads, self.head_dim)

        dk = (
            self.act(self.dk_proj(f_ij)).reshape(-1, self.num_heads, self.head_dim)
            if self.dk_proj is not None
            else None
        )
        dv = (
            self.act(self.dv_proj(f_ij)).reshape(-1, self.num_heads, self.head_dim)
            if self.dv_proj is not None
            else None
        )

        # propagate_type: (q: Tensor, k: Tensor, v: Tensor, dk: Tensor, dv: Tensor, r_ij: Tensor)
        dx = self.propagate(
            edge_index,
            edge_attr=e,
            q=q,
            k=k,
            v=v,
            dk=dk,
            dv=dv,
            r_ij=r_ij,
            size=None,
        )
        dx = dx.reshape(-1, self.hidden_channels)

        dx = self.o_proj(dx)

        return dx

    def message(self, edge_attr, q_i, k_j, v_j, dk, dv, r_ij):
        # attention mechanism
        if dk is None:
            attn = (q_i * k_j + edge_attr).sum(dim=-1)
        else:
            attn = (q_i * k_j * dk + edge_attr).sum(dim=-1)

        # attention activation function
        attn = self.attn_activation(attn) * self.cutoff(r_ij).unsqueeze(1)

        # value pathway
        if dv is not None:
            v_j = v_j * dv

        # update scalar features
        v_j = v_j * attn.unsqueeze(2)

        return v_j

    def aggregate(
        self,
        features: torch.Tensor,
        index: torch.Tensor,
        ptr: Optional[torch.Tensor],
        dim_size: Optional[int],
    ) -> torch.Tensor:
        x = scatter(features, index, dim=self.node_dim, dim_size=dim_size)
        return x

    def update(
        self, inputs: torch.Tensor
    ) -> torch.Tensor:
        return inputs
    
class MultiHeadCrossAttention(MessagePassing):
    def __init__(
        self,
        hidden_channels,
        num_rbf,
        distance_influence,
        num_heads,
        activation,
        attn_activation,
        cutoff_lower,
        cutoff_upper,
    ):
        super(MultiHeadCrossAttention, self).__init__(aggr="add", node_dim=0)
        assert hidden_channels % num_heads == 0, (
            f"The number of hidden channels ({hidden_channels}) "
            f"must be evenly divisible by the number of "
            f"attention heads ({num_heads})"
        )

        self.distance_influence = distance_influence
        self.num_heads = num_heads
        self.hidden_channels = hidden_channels
        self.head_dim = hidden_channels // num_heads

        self.act = act_class_mapping[activation]()
        self.attn_activation = act_class_mapping[attn_activation]()
        self.cutoff = CosineCutoff(cutoff_lower, cutoff_upper)

        self.q_proj = nn.Linear(hidden_channels, hidden_channels)
        self.k_proj = nn.Linear(hidden_channels, hidden_channels)
        self.v_proj = nn.Linear(hidden_channels, hidden_channels)
        self.o_proj = nn.Linear(hidden_channels, hidden_channels)

        self.edge_proj = nn.Linear(hidden_channels, hidden_channels)

        self.dk_proj = None
        if distance_influence in ["keys", "both"]:
            self.dk_proj = nn.Linear(num_rbf, hidden_channels)

        self.dv_proj = None
        if distance_influence in ["values", "both"]:
            self.dv_proj = nn.Linear(num_rbf, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        self.q_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.k_proj.weight)
        self.k_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.v_proj.weight)
        self.v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.edge_proj.weight)
        self.edge_proj.bias.data.fill_(0)
        if self.dk_proj:
            nn.init.xavier_uniform_(self.dk_proj.weight)
            self.dk_proj.bias.data.fill_(0)
        if self.dv_proj:
            nn.init.xavier_uniform_(self.dv_proj.weight)
            self.dv_proj.bias.data.fill_(0)

    def forward(self, x, x_cond, edge_index, edge_attr, r_ij, f_ij):
        q_cond = self.q_proj(x_cond).reshape(-1, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(-1, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(-1, self.num_heads, self.head_dim)

        e = self.edge_proj(edge_attr).reshape(-1, self.num_heads, self.head_dim)

        dk = (
            self.act(self.dk_proj(f_ij)).reshape(-1, self.num_heads, self.head_dim)
            if self.dk_proj is not None
            else None
        )
        dv = (
            self.act(self.dv_proj(f_ij)).reshape(-1, self.num_heads, self.head_dim)
            if self.dv_proj is not None
            else None
        )

        # propagate_type: (q: Tensor, k: Tensor, v: Tensor, dk: Tensor, dv: Tensor, r_ij: Tensor)
        dx_cond = self.propagate(
            edge_index,
            edge_attr=e,
            q=q_cond,
            k=k,
            v=v,
            dk=dk,
            dv=dv,
            r_ij=r_ij,
            size=None,
        )
        dx_cond = dx_cond.reshape(-1, self.hidden_channels)

        dx_cond = self.o_proj(dx_cond)

        return dx_cond
    

    def message(self, edge_attr, q_i, k_j, v_j, dk, dv, r_ij):
        # attention mechanism
        if dk is None:
            attn = (q_i * k_j + edge_attr).sum(dim=-1)
        else:
            attn = (q_i * k_j * dk + edge_attr).sum(dim=-1)

        # attention activation function
        attn = self.attn_activation(attn) * self.cutoff(r_ij).unsqueeze(1)

        # value pathway
        if dv is not None:
            v_j = v_j * dv

        # update scalar features
        v_j = v_j * attn.unsqueeze(2)

        return v_j

    def aggregate(
        self,
        features: torch.Tensor,
        index: torch.Tensor,
        ptr: Optional[torch.Tensor],
        dim_size: Optional[int],
    ) -> torch.Tensor:
        x = scatter(features, index, dim=self.node_dim, dim_size=dim_size)
        return x

    def update(
        self, inputs: torch.Tensor
    ) -> torch.Tensor:
        return inputs
    
class EdgeEmbeddingNetwork(nn.Module):
    def __init__(self, 
                 in_dim, 
                 num_rbf,
                 out_dim):
        super(EdgeEmbeddingNetwork, self).__init__()
        self.src_mapping = nn.Linear(in_dim, in_dim)
        self.tgt_mapping = nn.Linear(in_dim, in_dim)
        # self.node_mapping = nn.Linear(in_dim, in_dim)
        self.rbf_mapping = nn.Linear(num_rbf, in_dim)
        # self.out_mapping = MLP(in_dim * 2, in_dim)
        # self.out_mapping = MLP(in_dim * 3, out_dim)
        self.out_mapping = nn.Linear(in_dim * 3, out_dim)

    def reset_parameters(self):
        self.src_mapping.reset_parameters()
        self.tgt_mapping.reset_parameters()
        #self.node_mapping.reset_parameters()
        self.rbf_mapping.reset_parameters()
        self.out_mapping.reset_parameters()
    
    def forward(self, x, edge_index, rbf):
        src = edge_index[0]
        tgt = edge_index[1]
        rbf_map = self.rbf_mapping(rbf)
        x_src_map = self.src_mapping(x[src])
        x_tgt_map = self.tgt_mapping(x[tgt])
        # x_src_map = self.node_mapping(x[src])
        # x_tgt_map = self.node_mapping(x[tgt])
        out = self.out_mapping(torch.concat([x_src_map, x_tgt_map, rbf_map], dim=-1))
        # out = self.out_mapping(torch.concat([x_src_map * x_tgt_map, rbf_map], dim=-1))
        return out
    
class EdgeUpdateNetwork(nn.Module):
    def __init__(self, 
                 in_dim,
                 out_dim,
                 hid_dim=32):
        super(EdgeUpdateNetwork, self).__init__()
        self.node_mapping = nn.Linear(in_dim * 2, in_dim)
        self.mapping_in_a = nn.Linear(in_dim * 2, hid_dim)
        self.mapping_in_b = nn.Linear(in_dim * 2, hid_dim)
        self.out_mapping = nn.Linear(hid_dim ** 2, out_dim)

    def reset_parameters(self):
        self.node_mapping.reset_parameters()
        self.mapping_in_a.reset_parameters()
        self.mapping_in_b.reset_parameters()
        self.out_mapping.reset_parameters()
    
    def forward(self, x, edge_index, edge_attr):
        src, tgt = edge_index
        x_map = self.node_mapping(torch.concat([x[src], x[tgt]], dim=-1))
        a = self.mapping_in_a(torch.concat([x_map, edge_attr], dim=-1))
        b = self.mapping_in_b(torch.concat([x_map, edge_attr], dim=-1))
        a = a.unsqueeze(-1)
        b = b.unsqueeze(-2)
        outer = a * b
        outer = outer.reshape(edge_attr.shape[0], -1)

        out = self.out_mapping(outer)

        return out
    

class EdgeTriangularUpdate(nn.Module):
    def __init__(self, 
                 in_dim,
                 out_dim,
                 hidden_dim=32):
        super(EdgeTriangularUpdate, self).__init__()
        self.a_proj_g = nn.Linear(in_dim, hidden_dim)
        self.a_proj = nn.Linear(in_dim, hidden_dim)
        self.b_proj_g = nn.Linear(in_dim, hidden_dim)
        self.b_proj = nn.Linear(in_dim, hidden_dim)
        self.o_proj_g = nn.Linear(in_dim, out_dim)
        self.o_proj = nn.Linear(hidden_dim, out_dim)

        self.layer_norm = nn.LayerNorm(hidden_dim)

    def reset_parameters(self):
        # nn.init.xavier_uniform_(self.a_proj_g.weight)
        self.a_proj_g.weight.data.fill_(0)
        self.a_proj_g.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.a_proj.weight)
        self.a_proj.bias.data.fill_(0)
        # nn.init.xavier_uniform_(self.b_proj_g.weight)
        self.b_proj_g.weight.data.fill_(0)
        self.b_proj_g.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.b_proj.weight)
        self.b_proj.bias.data.fill_(0)
        # nn.init.xavier_uniform_(self.o_proj_g.weight)
        self.o_proj_g.weight.data.fill_(0)
        self.o_proj_g.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)
        self.layer_norm.reset_parameters()

    def forward(self, edge_index, edge_attr, batch):
        a = torch.sigmoid(self.a_proj_g(edge_attr)) * self.a_proj(edge_attr)
        b = torch.sigmoid(self.b_proj_g(edge_attr)) * self.b_proj(edge_attr)

        # mask = torch.sparse_coo_tensor(edge_index, torch.randn(edge_index.shape[1]).to(edge_index.device))

        # res1 = torch.hstack([torch.sparse.mm(torch.sparse_coo_tensor(edge_index, a_temp.squeeze()), torch.sparse_coo_tensor(edge_index, b_temp.squeeze()).T).sparse_mask(mask).coalesce().values().reshape(-1, 1) for a_temp, b_temp in zip(torch.split(a, 1, dim=-1), torch.split(b, 1, dim=-1))]) / self.dim
        # res2 = torch.hstack([torch.sparse.mm(torch.sparse_coo_tensor(edge_index, a_temp.squeeze()).T, torch.sparse_coo_tensor(edge_index, b_temp.squeeze())).sparse_mask(mask).coalesce().values().reshape(-1, 1) for a_temp, b_temp in zip(torch.split(a, 1, dim=-1), torch.split(b, 1, dim=-1))]) / self.dim

        dense_adj_matrix_a = torch_geometric.utils.to_dense_adj(edge_index, batch, a)
        dense_adj_matrix_b = torch_geometric.utils.to_dense_adj(edge_index, batch, b)
        mask_ = torch_geometric.utils.to_dense_adj(edge_index, batch)
        bool_mask = mask_.bool()

        temp1 = torch.einsum('bikm, bjkm -> bijm', dense_adj_matrix_a, dense_adj_matrix_b)
        res1 = temp1.masked_select(bool_mask.unsqueeze(-1))
        res1 = res1.reshape(-1, temp1.shape[-1])

        temp2 = torch.einsum('bkim, bkjm -> bijm', dense_adj_matrix_a, dense_adj_matrix_b)
        res2 = temp2.masked_select(bool_mask.unsqueeze(-1))
        res2 = res2.reshape(-1, temp2.shape[-1])

        o = res1 + res2

        o = self.layer_norm(o)

        output = torch.sigmoid(self.o_proj_g(edge_attr)) * self.o_proj(o)
        return output

class MLP(nn.Module):
    def __init__(self, 
                 in_dim,
                 out_dim,
                 num_layers=2):
        super(MLP, self).__init__()
        self.out = nn.ModuleList()
        for i in range(num_layers - 1):
            self.out.append(nn.Linear(in_dim, in_dim))
            self.out.append(nn.SiLU())
        self.out.append(nn.Linear(in_dim, out_dim))

    def reset_parameters(self):
        for m in self.out:
            try:
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0)
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0)
            except:
                continue

    def forward(self, feat):
        for m in self.out:
             feat = m(feat)
        return feat

class DistFlowMatchingNetwork(nn.Module):
    r"""The modified network architecture Based on the TorchMD equivariant Transformer architecture.

    Args:
        hidden_channels (int, optional): Hidden embedding size.
            (default: :obj:`128`)
        num_layers (int, optional): The number of attention layers.
            (default: :obj:`6`)
        num_rbf (int, optional): The number of radial basis functions :math:`\mu`.
            (default: :obj:`50`)
        rbf_type (string, optional): The type of radial basis function to use.
            (default: :obj:`"expnorm"`)
        trainable_rbf (bool, optional): Whether to train RBF parameters with
            backpropagation. (default: :obj:`True`)
        activation (string, optional): The type of activation function to use.
            (default: :obj:`"silu"`)
        attn_activation (string, optional): The type of activation function to use
            inside the attention mechanism. (default: :obj:`"silu"`)
        neighbor_embedding (bool, optional): Whether to perform an initial neighbor
            embedding step. (default: :obj:`True`)
        num_heads (int, optional): Number of attention heads.
            (default: :obj:`8`)
        distance_influence (string, optional): Where distance information is used inside
            the attention mechanism. (default: :obj:`"both"`)
        cutoff_lower (float, optional): Lower cutoff distance for interatomic interactions.
            (default: :obj:`0.0`)
        cutoff_upper (float, optional): Upper cutoff distance for interatomic interactions.
            (default: :obj:`5.0`)
        max_z (int, optional): Maximum atomic number. Used for initializing embeddings.
            (default: :obj:`100`)
    """

    def __init__(
        self,
        hidden_dim=128,
        num_layers=6,
        num_rbf=32,
        rbf_type="expnorm",
        trainable_rbf=True,
        activation="silu",
        attn_activation="silu",
        neighbor_embedding=True,
        num_heads=8,
        distance_influence="both",
        cutoff_lower=0.0,
        cutoff=5.0,
        max_embedding=100,
    ):
        super(DistFlowMatchingNetwork, self).__init__()

        assert distance_influence in ["keys", "values", "both", "none"]
        assert rbf_type in rbf_class_mapping, (
            f'Unknown RBF type "{rbf_type}". '
            f'Choose from {", ".join(rbf_class_mapping.keys())}.'
        )
        assert activation in act_class_mapping, (
            f'Unknown activation function "{activation}". '
            f'Choose from {", ".join(act_class_mapping.keys())}.'
        )
        assert attn_activation in act_class_mapping, (
            f'Unknown attention activation function "{attn_activation}". '
            f'Choose from {", ".join(act_class_mapping.keys())}.'
        )

        self.hidden_channels = hidden_dim
        self.num_layers = num_layers
        self.num_rbf = num_rbf
        self.rbf_type = rbf_type
        self.trainable_rbf = trainable_rbf
        self.activation = activation
        self.attn_activation = attn_activation
        self.neighbor_embedding = neighbor_embedding
        self.num_heads = num_heads
        self.distance_influence = distance_influence
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff
        self.max_z = max_embedding

        self.distance_expansion = rbf_class_mapping[rbf_type](
            self.cutoff_lower, self.cutoff_upper, self.num_rbf, True
        )

        self.node_encoding = NodeEncoding(self.hidden_channels, self.cutoff_lower, self.cutoff_upper, self.num_rbf, self.max_z)
        self.node_encoding_cond = NodeEncoding(self.hidden_channels, self.cutoff_lower, self.cutoff_upper, self.num_rbf, self.max_z)

        self.edge_embedding = EdgeEmbeddingNetwork(self.hidden_channels, self.num_rbf, self.hidden_channels)
        self.edge_embedding_cond = EdgeEmbeddingNetwork(self.hidden_channels, self.num_rbf, self.hidden_channels)

        self.attention_t = nn.ModuleList([MultiHeadAttention(self.hidden_channels, self.num_rbf, self.distance_influence, self.num_heads, self.activation, self.attn_activation, self.cutoff_lower, self.cutoff_upper) for i in range(self.num_layers)])

        self.attention_cond = nn.ModuleList([MultiHeadAttention(self.hidden_channels, self.num_rbf, self.distance_influence, self.num_heads, self.activation, self.attn_activation, self.cutoff_lower, self.cutoff_upper) for i in range(self.num_layers)])

        self.node_attn_layernorm = nn.ModuleList([nn.LayerNorm(self.hidden_channels) for i in range(self.num_layers)])
        self.node_attn_layernorm_cond = nn.ModuleList([nn.LayerNorm(self.hidden_channels) for i in range(self.num_layers)])

        self.cross_attention = nn.ModuleList([MultiHeadCrossAttention(self.hidden_channels, self.num_rbf, self.distance_influence, self.num_heads, self.activation, self.attn_activation, self.cutoff_lower, self.cutoff_upper) for i in range(self.num_layers)])
        self.cross_attention_cond = nn.ModuleList([MultiHeadCrossAttention(self.hidden_channels, self.num_rbf, self.distance_influence, self.num_heads, self.activation, self.attn_activation, self.cutoff_lower, self.cutoff_upper) for i in range(self.num_layers)])

        self.node_cross_attn_layernorm = nn.ModuleList([nn.LayerNorm(self.hidden_channels) for i in range(self.num_layers)])
        self.node_cross_attn_layernorm_cond = nn.ModuleList([nn.LayerNorm(self.hidden_channels) for i in range(self.num_layers)])

        self.edge_update = nn.ModuleList([EdgeUpdateNetwork(self.hidden_channels, self.hidden_channels) for i in range(self.num_layers)])
        self.edge_update_cond = nn.ModuleList([EdgeUpdateNetwork(self.hidden_channels, self.hidden_channels) for i in range(self.num_layers)])

        self.edge_attn_layernorm = nn.ModuleList([nn.LayerNorm(self.hidden_channels) for i in range(self.num_layers)])
        self.edge_attn_layernorm_cond = nn.ModuleList([nn.LayerNorm(self.hidden_channels) for i in range(self.num_layers)])

        self.edge_triangular_update = nn.ModuleList([EdgeTriangularUpdate(self.hidden_channels, self.hidden_channels) for i in range(self.num_layers)])
        self.edge_triangular_update_cond = nn.ModuleList([EdgeTriangularUpdate(self.hidden_channels, self.hidden_channels) for i in range(self.num_layers)])

        self.edge_tri_attn_layernorm = nn.ModuleList([nn.LayerNorm(self.hidden_channels) for i in range(self.num_layers)])
        self.edge_tri_attn_layernorm_cond = nn.ModuleList([nn.LayerNorm(self.hidden_channels) for i in range(self.num_layers)])

        self.flow_head_dist = EdgeUpdateNetwork(self.hidden_channels, 1)

        self.reset_parameters()

    def reset_parameters(self):
        self.node_encoding.reset_parameters()
        self.node_encoding_cond.reset_parameters()
        self.edge_embedding.reset_parameters()
        self.edge_embedding_cond.reset_parameters()
        self.distance_expansion.reset_parameters()
        for m in self.node_attn_layernorm:
            m.reset_parameters()
        for m in self.node_attn_layernorm_cond:
            m.reset_parameters()
        for m in self.node_cross_attn_layernorm:
            m.reset_parameters()
        for m in self.node_cross_attn_layernorm_cond:
            m.reset_parameters()
        for m in self.edge_attn_layernorm:
            m.reset_parameters()
        for m in self.edge_attn_layernorm_cond:
            m.reset_parameters()
        for m in self.edge_tri_attn_layernorm:
            m.reset_parameters()
        for m in self.edge_tri_attn_layernorm_cond:
            m.reset_parameters()
        for attn in self.attention_t:
            attn.reset_parameters()
        for attn in self.attention_cond:
            attn.reset_parameters()
        for attn in self.cross_attention:
            attn.reset_parameters()
        for attn in self.cross_attention_cond:
            attn.reset_parameters()
        for m in self.edge_update:
            m.reset_parameters()
        for m in self.edge_update_cond:
            m.reset_parameters()
        for m in self.edge_triangular_update:
            m.reset_parameters()
        for m in self.edge_triangular_update_cond:
            m.reset_parameters()
        self.flow_head_dist.reset_parameters()

    def forward(self, t, z, edge_index, dist_0, dist_T, dist_t, batch):
        
        rbf_0 = self.distance_expansion(dist_0)
        rbf_T = self.distance_expansion(dist_T)
        rbf_t = self.distance_expansion(dist_t)

        x_0 = self.node_encoding_cond(z, edge_index, dist_0, rbf_0, t)
        x_t = self.node_encoding(z, edge_index, dist_t, rbf_t, t)
        x_T = self.node_encoding_cond(z, edge_index, dist_T, rbf_T, t)

        # x_0 = self.node_encoding(z, edge_index, dist_0, rbf_0, t)
        # x_t = self.node_encoding(z, edge_index, dist_t, rbf_t, t)
        # x_T = self.node_encoding(z, edge_index, dist_T, rbf_T, t)

        edge_t = self.edge_embedding(x_t, edge_index, rbf_t)
        edge_0 = self.edge_embedding_cond(x_0, edge_index, rbf_0)
        edge_T = self.edge_embedding_cond(x_T, edge_index, rbf_T)

        # edge_t = self.edge_embedding(x_t, edge_index, rbf_t)
        # edge_0 = self.edge_embedding(x_0, edge_index, rbf_0)
        # edge_T = self.edge_embedding(x_T, edge_index, rbf_T)

        for i in range(self.num_layers):

            dx_t_cross_0 = self.cross_attention[i](x_t, x_0, edge_index, edge_t, dist_t, rbf_t)
            dx_t_cross_T = self.cross_attention[i](x_t, x_T, edge_index, edge_t, dist_t, rbf_t)

            dx_0_cross_t = self.cross_attention_cond[i](x_0, x_t, edge_index, edge_0, dist_0, rbf_0)
            dx_0_cross_T = self.cross_attention_cond[i](x_0, x_T, edge_index, edge_0, dist_0, rbf_0)

            dx_T_cross_0 = self.cross_attention_cond[i](x_T, x_0, edge_index, edge_T, dist_T, rbf_T)
            dx_T_cross_t = self.cross_attention_cond[i](x_T, x_t, edge_index, edge_T, dist_T, rbf_T)

            # dx_0_cross_t = self.cross_attention[i](x_0, x_t, edge_index, edge_0, dist_0, rbf_0)
            # dx_0_cross_T = self.cross_attention[i](x_0, x_T, edge_index, edge_0, dist_0, rbf_0)

            # dx_T_cross_0 = self.cross_attention[i](x_T, x_0, edge_index, edge_T, dist_T, rbf_T)
            # dx_T_cross_t = self.cross_attention[i](x_T, x_t, edge_index, edge_T, dist_T, rbf_T)

            x_0 = x_0 + dx_t_cross_0 + dx_T_cross_0
            x_t = x_t + dx_0_cross_t + dx_T_cross_t
            x_T = x_T + dx_0_cross_T + dx_t_cross_T

            # x_0 = self.node_cross_attn_layernorm[i](x_0)
            # x_t = self.node_cross_attn_layernorm[i](x_t)
            # x_T = self.node_cross_attn_layernorm[i](x_T)

            x_0 = self.node_cross_attn_layernorm_cond[i](x_0)
            x_t = self.node_cross_attn_layernorm[i](x_t)
            x_T = self.node_cross_attn_layernorm_cond[i](x_T)

            #----------------------

            dx_t = self.attention_t[i](x_t, edge_index, edge_t, dist_t, rbf_t)
            dx_0 = self.attention_cond[i](x_0, edge_index, edge_0, dist_0, rbf_0)
            dx_T = self.attention_cond[i](x_T, edge_index, edge_T, dist_T, rbf_T)

            # dx_t = self.attention_t[i](x_t, edge_index, edge_t, dist_t, rbf_t)
            # dx_0 = self.attention_t[i](x_0, edge_index, edge_0, dist_0, rbf_0)
            # dx_T = self.attention_t[i](x_T, edge_index, edge_T, dist_T, rbf_T)

            x_t = x_t + dx_t
            x_0 = x_0 + dx_0
            x_T = x_T + dx_T

            # x_0 = self.node_attn_layernorm[i](x_0)
            # x_t = self.node_attn_layernorm[i](x_t)
            # x_T = self.node_attn_layernorm[i](x_T)

            x_0 = self.node_attn_layernorm_cond[i](x_0)
            x_t = self.node_attn_layernorm[i](x_t)
            x_T = self.node_attn_layernorm_cond[i](x_T)

            #----------------------

            d_edge_t = self.edge_update[i](x_t, edge_index, edge_t)
            d_edge_0 = self.edge_update_cond[i](x_0, edge_index, edge_0)
            d_edge_T = self.edge_update_cond[i](x_T, edge_index, edge_T)

            # d_edge_t = self.edge_update[i](x_t, edge_index, rbf_t)
            # d_edge_0 = self.edge_update[i](x_0, edge_index, rbf_0)
            # d_edge_T = self.edge_update[i](x_T, edge_index, rbf_T)

            edge_0 = edge_0 + d_edge_0
            edge_t = edge_t + d_edge_t
            edge_T = edge_T + d_edge_T

            # edge_0 = self.edge_attn_layernorm[i](edge_0)
            # edge_t = self.edge_attn_layernorm[i](edge_t)
            # edge_T = self.edge_attn_layernorm[i](edge_T)

            edge_0 = self.edge_attn_layernorm_cond[i](edge_0)
            edge_t = self.edge_attn_layernorm[i](edge_t)
            edge_T = self.edge_attn_layernorm_cond[i](edge_T)

            #----------------------

            d_edge_0 = self.edge_triangular_update_cond[i](edge_index, edge_0, batch)
            d_edge_t = self.edge_triangular_update[i](edge_index, edge_t, batch)
            d_edge_T = self.edge_triangular_update_cond[i](edge_index, edge_T, batch)

            # d_edge_0 = self.edge_triangular_update[i](edge_index, edge_0, rbf_0, batch)
            # d_edge_t = self.edge_triangular_update[i](edge_index, edge_t, rbf_t, batch)
            # d_edge_T = self.edge_triangular_update[i](edge_index, edge_T, rbf_T, batch)

            edge_0 = edge_0 + d_edge_0
            edge_t = edge_t + d_edge_t
            edge_T = edge_T + d_edge_T

            # edge_0 = self.edge_tri_attn_layernorm[i](edge_0)
            # edge_t = self.edge_tri_attn_layernorm[i](edge_t)
            # edge_T = self.edge_tri_attn_layernorm[i](edge_T)

            edge_0 = self.edge_tri_attn_layernorm_cond[i](edge_0)
            edge_t = self.edge_tri_attn_layernorm[i](edge_t)
            edge_T = self.edge_tri_attn_layernorm_cond[i](edge_T)

        flow_dist = self.flow_head_dist(x_t, edge_index, edge_t).squeeze()

        return flow_dist

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"hidden_channels={self.hidden_channels}, "
            f"num_layers={self.num_layers}, "
            f"num_rbf={self.num_rbf}, "
            f"rbf_type={self.rbf_type}, "
            f"trainable_rbf={self.trainable_rbf}, "
            f"activation={self.activation}, "
            f"attn_activation={self.attn_activation}, "
            f"neighbor_embedding={self.neighbor_embedding}, "
            f"num_heads={self.num_heads}, "
            f"distance_influence={self.distance_influence}, "
            f"cutoff_lower={self.cutoff_lower}, "
            f"cutoff_upper={self.cutoff_upper})"
        )

class NeighborEmbedding(MessagePassing):

    def __init__(self, hidden_channels, num_rbf, cutoff_lower, cutoff_upper, max_z=100):
        super(NeighborEmbedding, self).__init__(aggr="add")
        self.embedding = nn.Embedding(max_z, hidden_channels)
        self.distance_proj = nn.Linear(num_rbf, hidden_channels)
        self.combine = nn.Linear(hidden_channels * 2, hidden_channels)
        self.cutoff = CosineCutoff(cutoff_lower, cutoff_upper)

        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()
        nn.init.xavier_uniform_(self.distance_proj.weight)
        nn.init.xavier_uniform_(self.combine.weight)
        self.distance_proj.bias.data.fill_(0)
        self.combine.bias.data.fill_(0)

    def forward(self, z, x, edge_index, edge_weight, edge_attr):
        # remove self loops
        mask = edge_index[0] != edge_index[1]
        if not mask.all():
            edge_index = edge_index[:, mask]
            edge_weight = edge_weight[mask]
            edge_attr = edge_attr[mask]

        C = self.cutoff(edge_weight)
        W = self.distance_proj(edge_attr) * C.view(-1, 1)

        x_neighbors = self.embedding(z)
        # propagate_type: (x: Tensor, W: Tensor)
        x_neighbors = self.propagate(edge_index, x=x_neighbors, W=W, size=None)
        x_neighbors = self.combine(torch.cat([x, x_neighbors], dim=1))
        return x_neighbors

    def message(self, x_j, W):
        return x_j * W


class GaussianSmearing(nn.Module):
    def __init__(self, cutoff_lower=0.0, cutoff_upper=5.0, num_rbf=50, trainable=True):
        super(GaussianSmearing, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.num_rbf = num_rbf
        self.trainable = trainable

        self.cutoff_fn = CosineCutoff(0, cutoff_upper)

        offset, coeff = self._initial_params()
        if trainable:
            self.register_parameter("coeff", nn.Parameter(coeff))
            self.register_parameter("offset", nn.Parameter(offset))
        else:
            self.register_buffer("coeff", coeff)
            self.register_buffer("offset", offset)

    def _initial_params(self):
        offset = torch.linspace(self.cutoff_lower, self.cutoff_upper, self.num_rbf)
        coeff = -0.5 / (offset[1] - offset[0]) ** 2
        return offset, coeff

    def reset_parameters(self):
        offset, coeff = self._initial_params()
        self.offset.data.copy_(offset)
        self.coeff.data.copy_(coeff)

    def forward(self, dist):
        dist = dist.unsqueeze(-1) - self.offset
        return torch.exp(self.coeff * torch.pow(dist, 2)) * self.cutoff_fn(dist)


class ExpNormalSmearing(nn.Module):
    def __init__(self, cutoff_lower=0.0, cutoff_upper=5.0, num_rbf=50, trainable=True):
        super(ExpNormalSmearing, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.num_rbf = num_rbf
        self.trainable = trainable

        self.cutoff_fn = CosineCutoff(0, cutoff_upper)
        self.alpha = 5.0 / (cutoff_upper - cutoff_lower)

        means, betas = self._initial_params()
        if trainable:
            self.register_parameter("means", nn.Parameter(means))
            self.register_parameter("betas", nn.Parameter(betas))
        else:
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)

    def _initial_params(self):
        # initialize means and betas according to the default values in PhysNet
        # https://pubs.acs.org/doi/10.1021/acs.jctc.9b00181
        start_value = torch.exp(
            torch.scalar_tensor(-self.cutoff_upper + self.cutoff_lower)
        )
        means = torch.linspace(start_value, 1, self.num_rbf)
        betas = torch.tensor(
            [(2 / self.num_rbf * (1 - start_value)) ** -2] * self.num_rbf
        )
        return means, betas

    def reset_parameters(self):
        means, betas = self._initial_params()
        self.means.data.copy_(means)
        self.betas.data.copy_(betas)

    def forward(self, dist):
        dist = dist.unsqueeze(-1)
        return self.cutoff_fn(dist) * torch.exp(
            -self.betas
            * (torch.exp(self.alpha * (-dist + self.cutoff_lower)) - self.means) ** 2
        )


class ShiftedSoftplus(nn.Module):
    def __init__(self):
        super(ShiftedSoftplus, self).__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift


class CosineCutoff(nn.Module):
    def __init__(self, cutoff_lower=0.0, cutoff_upper=5.0):
        super(CosineCutoff, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper

    def forward(self, distances):
        if self.cutoff_lower > 0:
            cutoffs = 0.5 * (
                torch.cos(
                    math.pi
                    * (
                        2
                        * (distances - self.cutoff_lower)
                        / (self.cutoff_upper - self.cutoff_lower)
                        + 1.0
                    )
                )
                + 1.0
            )
            # remove contributions below the cutoff radius
            cutoffs = cutoffs * (distances < self.cutoff_upper).float()
            cutoffs = cutoffs * (distances > self.cutoff_lower).float()
            return cutoffs
        else:
            cutoffs = 0.5 * (torch.cos(distances * math.pi / self.cutoff_upper) + 1.0)
            # remove contributions beyond the cutoff radius
            cutoffs = cutoffs * (distances < self.cutoff_upper).float()
            return cutoffs



rbf_class_mapping = {"gauss": GaussianSmearing, "expnorm": ExpNormalSmearing}

act_class_mapping = {
    "ssp": ShiftedSoftplus,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
}

class TimestepEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super(TimestepEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
    def forward(self, time):
        device = time.device
        half_dim = self.embedding_dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)

        embeddings = time * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
    
class ODEWrapper(torch.nn.Module):
    def __init__(self, 
                 dynamics):
        super(ODEWrapper, self).__init__()
        self.dynamics = dynamics

    def forward(self, x, edge_index, dist_reactant, dist_product, dist_lin_interp, batch, step_size=0.05):
        def partial_func(t, dist_pred):
            return self.dynamics(torch.ones_like(x, dtype=torch.float32).unsqueeze(-1) * t, x, edge_index, dist_reactant, dist_product, dist_pred, batch)
        pred = odeint_adjoint(partial_func, dist_lin_interp, torch.tensor([0.0, 1.0]), method="midpoint", options=dict(step_size=step_size), adjoint_rtol=1e-4, adjoint_atol=1e-4, rtol=1e-4, atol=1e-4, adjoint_params=self.dynamics.parameters())[-1]
        return pred
    
class ODEWrapper2(torch.nn.Module):
    def __init__(self, 
                 dynamics):
        super(ODEWrapper2, self).__init__()
        self.dynamics = dynamics

    def forward(self, x, edge_index, dist_reactant, dist_product, dist_lin_interp, batch, step_size=0.05):
        def partial_func(t, dist_pred):
            # return self.dynamics(torch.ones_like(x, dtype=torch.float32).unsqueeze(-1) * t, x, edge_index, dist_reactant, dist_product, dist_pred, batch) - (dist_reactant + dist_product) / 2
            return self.dynamics(torch.ones_like(x, dtype=torch.float32).unsqueeze(-1) * t, x, edge_index, dist_reactant, dist_product, dist_pred, batch) / (1 - t)
        pred = odeint_adjoint(partial_func, dist_lin_interp, torch.tensor([0.0, 1.0]), method="midpoint", options=dict(step_size=step_size), adjoint_rtol=1e-4, adjoint_atol=1e-4, rtol=1e-4, atol=1e-4, adjoint_params=self.dynamics.parameters())[-1]
        return pred