import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.checkpoint import checkpoint

def identity(x, *args, **kwargs):
    return x

def get_act(activation):
    if activation == "gelu":
        return F.gelu
    if activation == "relu":
        return F.relu
    return None

def get_EF(input_size, dim, bias=True):
    lin = nn.Linear(input_size, dim, bias)
    torch.nn.init.kaiming_normal_(lin.weight, mode='fan_in', nonlinearity='relu')
    return lin

class Residual(nn.Module):
    def __init__(self, fn, input_channels=0, output_channels=0):
        super(Residual, self).__init__()
        self.fn = fn
        self.resample = nn.Linear(input_channels, output_channels) if input_channels != output_channels else None
        self.norm = nn.LayerNorm(output_channels)

    def forward(self, tensor, **kwargs):
        tensor = tensor + self.fn(tensor, **kwargs)
        tensor = self.norm(tensor)
        return tensor

class ProjectInOut(nn.Module):
    def __init__(self, fn, dim_in, dim_out, project_out=True):
        super(ProjectInOut, self).__init__()
        self.fn = fn
        self.project_in = nn.Linear(dim_in, dim_out)
        self.project_out = nn.Linear(dim_out, dim_in) if project_out else identity

    def forward(self, tensor, **kwargs):
        tensor = self.project_in(tensor)
        tensor = self.fn(tensor, **kwargs)
        tensor = self.project_out(tensor)
        return tensor

class FeedForward(nn.Module):
    def __init__(self, input_channels, output_channels, ff_dim, dropout, activation="relu"):
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(input_channels, ff_dim)
        self.w_2 = nn.Linear(ff_dim, output_channels)
        self.activation = get_act(activation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tensor, **kwargs):
        tensor = self.w_1(tensor)
        tensor = self.activation(tensor)
        tensor = self.dropout(tensor)
        tensor = self.w_2(tensor)
        return tensor

class LinearAttentionHead(nn.Module):
    def __init__(self, dim, dropout, E_proj, F_proj, full_attention=False):
        super(LinearAttentionHead, self).__init__()
        self.E = E_proj
        self.F = F_proj
        self.dim = dim
        self.dropout = nn.Dropout(dropout)
        self.P_bar = None
        self.full_attention = full_attention
        self.is_proj_tensor = isinstance(E_proj, torch.Tensor)

    def forward(self, Q, K, V, **kwargs):
        input_mask = kwargs["input_mask"] if "input_mask" in kwargs else None
        embeddings_mask = kwargs["embeddings_mask"] if "embeddings_mask" in kwargs else None

        if input_mask is not None:
            mask = input_mask[:,:,None]
            K = K.masked_fill_(~mask, 0.0)
            V = V.masked_fill_(~mask, 0.0)
            del mask

        if embeddings_mask is not None:
            mask = embeddings_mask[:,:,None]
            Q = Q.masked_fill_(~mask, 0.0)
            del mask

        K = K.transpose(1,2)
        if not self.full_attention:
            if self.is_proj_tensor:
                self.E = self.E.to(K.device)
                K = torch.matmul(K, self.E)
            else:
                K = self.E(K)
        Q = torch.matmul(Q, K)

        P_bar = Q/torch.sqrt(torch.tensor(self.dim).type(Q.type())).to(Q.device)
        P_bar = P_bar.softmax(dim=-1)

        if "visualize" in kwargs and kwargs["visualize"] == True:
            self.P_bar = P_bar

        P_bar = self.dropout(P_bar)

        if not self.full_attention:
            V = V.transpose(1,2)
            if self.is_proj_tensor:
                self.F = self.F.to(V.device)
                V = torch.matmul(V, self.F)
            else:
                V = self.F(V)
            V = V.transpose(1,2)
        out_tensor = torch.matmul(P_bar, V)

        return out_tensor

class Gene2VecPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        gene2vec_weight = np.load('../data/embedding _TruncatedSVDd.npy')
        gene2vec_weight = np.concatenate((gene2vec_weight, np.zeros((1, gene2vec_weight.shape[1]))), axis=0)
        gene2vec_weight = torch.from_numpy(gene2vec_weight)
        self.emb = nn.Embedding.from_pretrained(gene2vec_weight)
    def forward(self, x):
        t = torch.arange(x.shape[1], device=x.device)
        return self.emb(t)

class Always(nn.Module):
    def __init__(self, val):
        super().__init__()
        self.val = val

    def forward(self, *args, **kwargs):
        return self.val

class MHAttention(nn.Module):
    def __init__(self, input_size, dim, channels, dim_k, nhead, dropout, checkpoint_level,
            parameter_sharing, E_proj, F_proj, full_attention, w_o_intermediate_dim=None, method="learnable"):
        super(MHAttention, self).__init__()
        self.heads = nn.ModuleList()
        self.input_size = input_size
        self.dim_k = dim_k
        self.channels = channels
        self.checkpoint_level = checkpoint_level
        self.w_o_intermediate_dim = w_o_intermediate_dim
        if parameter_sharing != "layerwise":
            E_proj = get_EF(input_size, dim_k, method, dim)
            F_proj = get_EF(input_size, dim_k, method, dim) if parameter_sharing == "none" or parameter_sharing == "headwise" else E_proj
        self.to_q = nn.ModuleList()
        self.to_k = nn.ModuleList()
        self.to_v = nn.ModuleList()

        for _ in range(nhead):
            if parameter_sharing == "none":
                E_proj = get_EF(input_size, dim_k, method, dim)
                F_proj = get_EF(input_size, dim_k, method, dim)
            attn = LinearAttentionHead(dim, dropout, E_proj, F_proj, full_attention)
            self.heads.append(attn)
            self.to_q.append(nn.Linear(channels, dim, bias=False))
            self.to_k.append(nn.Linear(channels, dim, bias=False))
            self.to_v.append(nn.Linear(channels, dim, bias=False))
        if w_o_intermediate_dim is None:
            self.w_o = nn.Linear(dim*nhead, channels)
        else:
            self.w_o_1 = nn.Linear(dim*nhead, w_o_intermediate_dim)
            self.w_o_2 = nn.Linear(w_o_intermediate_dim, channels)
        self.mh_dropout = nn.Dropout(dropout)


    def forward(self, tensor, **kwargs):
        head_outputs = []
        for index, head in enumerate(self.heads):
            Q = self.to_q[index](tensor)
            K = self.to_k[index](tensor)
            V = self.to_v[index](tensor)
            head_outputs.append(head(Q,K,V,**kwargs))
        out = torch.cat(head_outputs, dim=-1)
        if self.w_o_intermediate_dim is None:
            out = self.w_o(out)
        else:
            out = self.w_o_1(out)
            out = self.w_o_2(out)
        out = self.mh_dropout(out)
        return out
def exists(val):
    return val is not None
class Linformer(nn.Module):
    def __init__(self, input_size, channels, dim_k, dim_ff=256, dim_d=None, dropout_ff=0.15, nhead=4, depth=1, dropout=0, activation="relu", checkpoint_level="C0", parameter_sharing="layerwise", k_reduce_by_layer=0, full_attention=False,  w_o_intermediate_dim=None, method="learnable", ff_intermediate=None):
        super(Linformer, self).__init__()
        layers = nn.ModuleList()
        self.input_size = input_size
        self.channels = channels
        self.checkpoint_level = checkpoint_level
        self.depth = depth
        self.nhead = nhead

        head_dim = dim_d
        E_proj = get_EF(input_size, dim_k)
        get_attn = lambda attn_channels, curr_dim_k: MHAttention(input_size, head_dim, attn_channels, curr_dim_k, nhead, dropout, checkpoint_level, parameter_sharing, E_proj, E_proj, full_attention, w_o_intermediate_dim, method=method)
        get_ff = lambda input_channels, output_channels: FeedForward(input_channels, output_channels, dim_ff, dropout_ff, activation)

        for index in range(depth):
            input_channels = channels
            output_channels = channels
            attn_layer = get_attn(input_channels, max(1, dim_k - index*k_reduce_by_layer))
            ff_layer = get_ff(input_channels, output_channels)
            attn_layer, ff_layer = map(lambda res_ch_in, res_ch_out, fn: Residual(fn, res_ch_in, res_ch_out), (input_channels, input_channels), (input_channels, output_channels), (attn_layer, ff_layer))
            layers.extend([attn_layer, ff_layer])
        self.seq = layers

    def forward(self, tensor, **kwargs):
        for layer in self.seq:
            tensor = layer(tensor, **kwargs)
        return tensor

class LinformerLM(nn.Module):
    def __init__(self, num_tokens, input_size, channels,
                       dim_k=64, dim_ff=1024, dim_d=None,
                       dropout_ff=0, dropout_tokens=0.0, nhead=4, depth=2, ff_intermediate=None,
                       dropout=0, activation="relu", checkpoint_level="C0",
                       parameter_sharing="layerwise", k_reduce_by_layer=1, full_attention=False,
                       w_o_intermediate_dim=None,  method="learnable", peak_svd_embedding = True):
        super(LinformerLM, self).__init__()
        emb_dim = channels

        self.input_size = input_size
        self.to_token_emb = nn.Embedding(num_tokens, emb_dim)
        self.linformer = Linformer(input_size, channels, dim_k=dim_k,
                                   dim_ff=dim_ff, dim_d=dim_d, dropout_ff=dropout_ff,
                                   nhead=nhead, depth=depth, dropout=dropout, ff_intermediate=ff_intermediate,
                                   activation=activation, checkpoint_level=checkpoint_level, parameter_sharing=parameter_sharing,
                                   k_reduce_by_layer=k_reduce_by_layer, full_attention=full_attention,
                                   w_o_intermediate_dim=w_o_intermediate_dim, method=method)
        self.to_logits = nn.Linear(emb_dim, num_tokens)
        self.norm = nn.LayerNorm(emb_dim)
        self.dropout_tokens = nn.Dropout(dropout_tokens)
        if peak_svd_embedding:
            self.pos_emb = Gene2VecPositionalEmbedding(emb_dim, input_size)
        else:
            self.pos_emb = torch.zeros_like

    def forward(self, tensor, **kwargs):
        tensor = self.to_token_emb(tensor).cuda()
        tensor += self.pos_emb(tensor)
        tensor = self.dropout_tokens(tensor).cuda()
        tensor = self.linformer(tensor, **kwargs).cuda()
        tensor = self.norm(tensor)
        if exists(self.to_logits):
            tensor = self.to_logits(tensor).cuda()
            return tensor

        return tensor @ self.to_logits(tensor).cuda()

