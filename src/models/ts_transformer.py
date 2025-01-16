from typing import Optional, Any
import math
import logging
from typing import List, Optional
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.modules import MultiheadAttention, Linear, BatchNorm1d, TransformerEncoderLayer
try:
    from apex.normalization import FusedLayerNorm as _FusedLayerNorm

    has_fused_layernorm = True

    class FusedLayerNorm(_FusedLayerNorm):
        @torch.jit.unused
        def forward(self, x):
            if not x.is_cuda:
                return super().forward(x)
            else:
                with torch.cuda.device(x.device):
                    return super().forward(x)

except ImportError:
    has_fused_layernorm = False


def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True, export=False):
    if torch.jit.is_scripting():
        export = True
    if not export and torch.cuda.is_available() and has_fused_layernorm:
        return FusedLayerNorm(normalized_shape, eps, elementwise_affine)
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)


class Fp32LayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = F.layer_norm(
            input.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)


def model_factory(config, data):
    task = config['task']
    feat_dim = data.feature_df.shape[1]  # dimensionality of data features
    # data windowing is used when samples don't have a predefined length or the length is too long
    max_seq_len = config['data_window_len'] if config['data_window_len'] is not None else config['max_seq_len']
    if max_seq_len is None:
        try:
            max_seq_len = data.max_seq_len
        except AttributeError as x:
            print(
                "Data class does not define a maximum sequence length, so it must be defined with the script argument `max_seq_len`")
            raise x

    if (task == "imputation") or (task == "transduction"):
        return TSTransformerEncoder(feat_dim, max_seq_len, config['d_model'], config['num_heads'],
                                    config['num_layers'], config['dim_feedforward'], dropout=config['dropout'],
                                    pos_encoding=config['pos_encoding'], activation=config['activation'],
                                    norm=config['normalization_layer'], freeze=config['freeze'])

    if (task == "classification") or (task == "regression"):
        num_labels = len(data.class_names) if task == "classification" else data.labels_df.shape[
            1]  # dimensionality of labels
        return TSTransformerEncoderClassiregressor(feat_dim, max_seq_len, config['d_model'],
                                                    config['num_heads'],
                                                    config['num_layers'], config['dim_feedforward'],
                                                    num_classes=num_labels,
                                                    dropout=config['dropout'], pos_encoding=config['pos_encoding'],
                                                    activation=config['activation'],
                                                    norm=config['normalization_layer'], freeze=config['freeze'])
    else:
        raise ValueError("Model class for task '{}' does not exist".format(task))


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise ValueError("activation should be relu/gelu, not {}".format(activation))


# From https://github.com/pytorch/examples/blob/master/word_language_model/model.py
class FixedPositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=1024).
    """

    def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0):
        super(FixedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # positional encoding
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = scale_factor * pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)  # this stores the variable in the state_dict (used for non-trainable variables)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class dba_attention(nn.Module):
    def __init__(self, d_model, proj_len, nhead, dropout=0.1):
        super(dba_attention, self).__init__()
        self.middle_dim = 24
        self.proj_dim = proj_len
        self.d_model = d_model
        self.nhead = nhead

        self.pq_proj = nn.Linear(d_model, self.middle_dim)
        self.q_proj = nn.Linear(d_model, self.middle_dim)
        self.q_len_proj = nn.Linear(d_model, self.proj_dim * nhead)

        self.pk_proj = nn.Linear(d_model, self.middle_dim)
        self.pv_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, self.middle_dim)
        self.v_proj = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)

        self.head_dim = self.middle_dim // nhead
        self.head_dim_v = d_model // nhead
        self.phead_dim = self.middle_dim // nhead
        self.scaling = self.head_dim ** -0.5
        self.pscaling = self.phead_dim ** -0.5

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.self_attn_layer_norm1 = LayerNorm(d_model)

    def forward(self, x, px, mask):
        bsz = x.size(1)
        len = x.size(0)

        pk = self.pk_proj(x).view(len, bsz * self.nhead, self.head_dim)
        pv = self.pv_proj(x).view(len, bsz * self.nhead, self.head_dim_v)

        pk = pk.permute(1, 2, 0)
        # N x B*H x K -> B*H x N x K
        pv = pv.transpose(0, 1)

        pq = self.pq_proj(px).view(self.proj_dim, bsz * self.nhead, self.head_dim)
        # L x B*H x K -> B*H x L x K
        pq = pq.transpose(0, 1) * self.pscaling
        # B*H x L x N
        pqk = torch.bmm(pq, pk)

        if mask is not None:
            pqk = pqk.view(bsz, self.nhead, self.proj_dim, len)
            pqk = pqk.masked_fill(mask.unsqueeze(1).unsqueeze(2).to(torch.bool), float("-inf"))
            pqk = pqk.view(bsz * self.nhead, self.proj_dim, len)

        pqk = F.softmax(pqk, dim=-1)
        pqk = self.dropout1(pqk)
        # B*H x L x K
        pc = torch.bmm(pqk, pv)
        # B*H x L x K -> L x B*H x K -> L x B x D
        pc = pc.transpose(0, 1).contiguous().view(self.proj_dim, bsz, self.d_model)

        q_len = self.q_len_proj(x).view(-1, bsz * self.nhead, self.proj_dim).transpose(0, 1)
        k = self.k_proj(pc).view(-1, bsz * self.nhead, self.head_dim).transpose(0, 1)
        v = self.v_proj(pc).view(-1, bsz * self.nhead, self.head_dim_v).transpose(0, 1)
        q = self.q_proj(pc)
        q = q * self.scaling
        q = q.contiguous().view(-1, bsz * self.nhead, self.head_dim).transpose(0, 1)

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = attn_weights.view(bsz, self.nhead, self.proj_dim, -1)
        attn_weights = attn_weights.view(bsz * self.nhead, self.proj_dim, -1)

        attn_weights_float = F.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = self.dropout2(attn_weights)

        attn = torch.bmm(attn_probs, v)

        attn = attn.transpose(0, 1).contiguous().view(-1, bsz, self.d_model)
        attn = self.self_attn_layer_norm1(attn)
        attn = attn.view(-1, bsz * self.nhead, self.head_dim_v).transpose(0, 1)

        q_len = F.softmax(q_len, dim=-1)
        q_len = self.dropout3(q_len)
        attn = torch.bmm(q_len, attn)

        attn = attn.transpose(0, 1).contiguous().view(-1, bsz, self.d_model)
        attn = self.out_proj(attn)
        return attn, pc


class LearnablePositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Each position gets its own embedding
        # Since indices are always 0 ... max_len, we don't have to do a look-up
        self.pe = nn.Parameter(torch.empty(max_len, 1, d_model))  # requires_grad automatically set to True
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def get_pos_encoder(pos_encoding):
    if pos_encoding == "learnable":
        return LearnablePositionalEncoding
    elif pos_encoding == "fixed":
        return FixedPositionalEncoding

    raise NotImplementedError("pos_encoding should be 'learnable'/'fixed', not '{}'".format(pos_encoding))

class dba_attention_layer(nn.modules.Module):
    r"""This transformer encoder layer block is made up of self-attn and feedforward network.
    It differs from TransformerEncoderLayer in torch/nn/modules/transformer.py in that it replaces LayerNorm
    with BatchNorm.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    """

    def __init__(self, d_model,  proj_len, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(dba_attention_layer, self).__init__()
        self.self_attn = dba_attention(d_model, proj_len, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        # gain = 1.0 / math.sqrt(2.0)
        # nn.init.xavier_uniform_(self.linear1.weight, gain=gain)
        # nn.init.xavier_uniform_(self.linear2.weight, gain=gain)

        self.norm1 = LayerNorm(d_model)  # normalizes each feature across batch samples and time steps
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)



    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(dba_attention_layer, self).__setstate__(state)

    def forward(self, src, px, src_key_padding_mask):
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2, px = self.self_attn(src, px, src_key_padding_mask)
        src = src + self.dropout1(src2)  # (seq_len, batch_size, d_model)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)  # (seq_len, batch_size, d_model)
        src = self.norm2(src)
        return src, px

class TSTransformerEncoder(nn.Module):

    def __init__(self, feat_dim, max_len, d_model, n_heads, num_layers, dim_feedforward, dropout=0.1,
                 pos_encoding='fixed', activation='gelu', norm='BatchNorm', freeze=False, attention_dropout=0.1):
        super(TSTransformerEncoder, self).__init__()

        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads

        self.project_inp = nn.Linear(feat_dim, d_model)
        self.pos_enc = get_pos_encoder(pos_encoding)(d_model, dropout=dropout * (1.0 - freeze), max_len=max_len)

        self.proj_len = 16

        self.px = nn.Parameter(torch.Tensor(self.proj_len, self.embedding_dim))
        nn.init.normal_(self.projected_embeddings, mean=0., std=self.embedding_dim ** -0.5)
        self.encoder_layers = nn.ModuleList([])
        self.encoder_layers.extend(
            [dba_attention_layer(d_model, self.proj_len, self.n_heads, dim_feedforward, attention_dropout * (1.0 - freeze),
                                                    activation=activation) for i in range(num_layers)]
        )

        self.output_layer = nn.Linear(d_model, feat_dim)
        self.act = _get_activation_fn(activation)
        self.dropout1 = nn.Dropout(dropout)
        self.feat_dim = feat_dim

    def forward(self, X, padding_masks):
        """
        Args:
            X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, seq_length, feat_dim)
        """
        px = self.px
        # permute because pytorch convention for transformers is [seq_length, batch_size, feat_dim]. padding_masks [batch_size, feat_dim]
        inp = X.permute(1, 0, 2)
        inp = self.project_inp(inp) * math.sqrt(
            self.d_model)  # [seq_length, batch_size, d_model] project input vectors to d_model dimensional space
        inp = self.pos_enc(inp)  # add positional encoding
        # NOTE: logic for padding masks is reversed to comply with definition in MultiHeadAttention, TransformerEncoderLayer
        # output = self.transformer_encoder(inp, src_key_padding_mask=~padding_masks)  # (seq_length, batch_size, d_model)
        x = inp
        for encoder_layer in self.encoder_layers:
            x, px = encoder_layer(x, px, ~padding_masks)
        output = self.act(x)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = output.permute(1, 0, 2)  # (batch_size, seq_length, d_model)
        output = self.dropout1(output)
        # Most probably defining a Linear(d_model,feat_dim) vectorizes the operation over (seq_length, batch_size).
        output = self.output_layer(output)  # (batch_size, seq_length, feat_dim)

        return output


class TSTransformerEncoderClassiregressor(nn.Module):
    """
    Simplest classifier/regressor. Can be either regressor or classifier because the output does not include
    softmax. Concatenates final layer embeddings and uses 0s to ignore padding embeddings in final output layer.
    """

    def __init__(self, feat_dim, max_len, d_model, n_heads, num_layers, dim_feedforward, num_classes,
                 dropout=0.1, pos_encoding='fixed', activation='gelu', norm='BatchNorm', freeze=False, attention_dropout=0.1):
        super(TSTransformerEncoderClassiregressor, self).__init__()

        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads

        self.project_inp = nn.Linear(feat_dim, d_model)
        self.pos_enc = get_pos_encoder(pos_encoding)(d_model, dropout=dropout * (1.0 - freeze), max_len=max_len)

        self.proj_len = 16

        self.px = nn.Parameter(torch.Tensor(self.proj_len, d_model))
        nn.init.normal_(self.px, mean=0., std=d_model ** -0.5)

        self.encoder_layers = nn.ModuleList([])
        self.encoder_layers.extend(
            [dba_attention_layer(d_model, self.proj_len, self.n_heads, dim_feedforward, dropout * (1.0 - freeze),
                                 activation=activation) for i in range(num_layers)])

        self.act = _get_activation_fn(activation)

        self.dropout1 = nn.Dropout(dropout)

        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.output_layer = self.build_output_module(d_model, max_len, num_classes)

        self.dropout_module = nn.Dropout(dropout)

    def build_output_module(self, d_model, max_len, num_classes):
        output_layer = nn.Linear(d_model * max_len, num_classes)
        return output_layer

    def forward(self, X, padding_masks):
        """
        Args:
            X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, num_classes)
        """
        px=self.px
        px = px.unsqueeze(1).expand(px.size(0), X.size(0), self.d_model)
        px = self.dropout_module(px)
        # permute because pytorch convention for transformers is [seq_length, batch_size, feat_dim]. padding_masks [batch_size, feat_dim]
        inp = X.permute(1, 0, 2)
        inp = self.project_inp(inp) * math.sqrt(
            self.d_model)  # [seq_length, batch_size, d_model] project input vectors to d_model dimensional space
        inp = self.pos_enc(inp)  # add positional encoding
        # NOTE: logic for padding masks is reversed to comply with definition in MultiHeadAttention, TransformerEncoderLayer
        x = inp
        for encoder_layer in self.encoder_layers:
            x, px = encoder_layer(x, px, ~padding_masks)
        output = self.act(x)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = output.permute(1, 0, 2)  # (batch_size, seq_length, d_model)
        output = self.dropout1(output)

        # Output
        output = output * padding_masks.unsqueeze(-1)  # zero-out padding embeddings
        output = output.reshape(output.shape[0], -1)  # (batch_size, seq_length * d_model)
        output = self.output_layer(output)  # (batch_size, num_classes)

        return output
