from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, LongTensor

from rotary_embedding_torch import RotaryEmbedding

    
class RoPETransformer(nn.Module):
    """
    Decoder with RoPE.
    """
    def __init__(self, Attn_Module: nn.Module, args):
        super().__init__()
        self.args = args
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(args.vocab_size, args.n_embd),
            dropout = nn.Dropout(args.dp),
            h = nn.ModuleList(Block(Attn_Module, args) for _ in range(args.n_layer)),
            ln_f = nn.LayerNorm(args.n_embd) if args.if_ln else nn.Identity()
        ))
        self.n_layer = args.n_layer
        self.lm_head = nn.Linear(args.n_embd, args.vocab_size, bias=False)
        self.lm_head.g_scale = 1.
        
        for id, m in self.named_modules():
            if isinstance(m, (nn.Linear,)):
                nn.init.normal_(m.weight, std=0.02)
                if 'proj' in id:
                    m.weight.data /= (2 * self.n_layer)**0.5
                if 'head' in id:
                    m.weight.data /= self.args.n_embd**0.5
            if isinstance(m, (nn.Embedding,)) and 'wte' in id:
                nn.init.normal_(m.weight, std=0.02)
        if args.weight_tying is True:
            self.lm_head.weight = self.transformer.wte.weight

    def forward(self, idx: LongTensor) -> Tensor:
        device = idx.device
        b, t = idx.size()

        tok_emb = self.transformer.wte(idx)
        x = self.transformer.dropout(tok_emb)
        for block in self.transformer.h:
            x, _ = block(x)
        x = self.transformer.ln_f(x)
        # logits = self.scale * self.lm_head(x)
        logits = self.lm_head(x)

        return logits

    @torch.inference_mode()
    def record(self, idx: LongTensor) -> Tensor:
        qkv_list = []
        input_list = []
        output_list = []
        device = idx.device
        b, t = idx.size()

        tok_emb = self.transformer.wte(idx)
        x = self.transformer.dropout(tok_emb)
        for block in self.transformer.h:
            input_list.append(x)
            x, (q, k, v) = block(x)
            qkv_list.append((q, k, v))
            output_list.append(x)
        x = self.transformer.ln_f(x)
        # logits = self.scale * self.lm_head(x)
        logits = self.lm_head(x)

        return logits, (qkv_list, input_list, output_list)

    @torch.inference_mode()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, deterministic=True):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.args.block_size else idx[:, -self.args.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if deterministic:
                idx_next = logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
    

########################################
########## Layers
########################################
class Block(nn.Module):
    """
    Standard Decoder Block.
    """
    def __init__(self, Attn_Module: nn.Module, args):
        super().__init__()
        # self.mu = args.mu
        self.ln_1 = nn.LayerNorm(args.n_embd) if args.if_ln else nn.Identity()
        self.attn = Attn_Module(args)
        self.ln_2 = nn.LayerNorm(args.n_embd) if args.if_ln else nn.Identity()
        if args.act_name.lower() == 'swiglu':
            self.mlp = LlamaMLP(args.n_embd, args.n_embd, 8, args.widen_factor)
        else:
            self.mlp = MLP(args)
        return

    def forward(self, x: Tensor) -> Tensor:
        y, (q, k, v) = self.attn(self.ln_1(x))
        x = x + y
        x = x + self.mlp(self.ln_2(x))
        return x, (q, k, v)
    

class RoPEFlashAttention(nn.Module):
    """
    MultiHead Attention with RoPE with PyTorch's implementation of flash attention.
    """
    def __init__(self, args, current_d: int = 1):
        super().__init__()
        self.n_embd = args.n_embd
        self.current_d = current_d
        self.to_qkv = nn.Linear(self.current_d * self.n_embd, 3 * self.current_d * self.n_embd, bias=False)
        self.o = nn.Linear(self.current_d * self.n_embd, self.n_embd, bias=False)
        self.o.g_scale = self.n_embd**(-1)
        self.dp = args.dp
        self.resid_dropout = nn.Dropout(args.dp)
        self.n_head = args.n_head
        self.power = - 0.5 * (1. + args.s)
        self.is_causal = False if 'encoder' in args.model_name else True
        # self.rotary_emb = RoPE(self.n_embd // self.n_head)
        self.rotary_emb = RotaryEmbedding(dim=self.current_d * self.n_embd // args.n_head)

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.size()
        q, k, v = self.to_qkv(x).split(self.current_d * self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Apply rotary embeddings
        q = self.rotary_emb.rotate_queries_or_keys(q)
        k = self.rotary_emb.rotate_queries_or_keys(k)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, scale=(q.size(-1))**self.power, dropout_p=self.dp if self.training else 0, is_causal=self.is_causal)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.o(y))
        return y, (q, k, v)


class LlamaMLP(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        """
        Initialize the FeedForward module.

        Args:
            dim (int): Input dimension.
            hidden_dim (int): Hidden dimension of the feedforward layer.
            multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
            ffn_dim_multiplier (float, optional): Custom multiplier for hidden dimension. Defaults to None.

        Attributes:
            w1 (ColumnParallelLinear): Linear transformation for the first layer.
            w2 (RowParallelLinear): Linear transformation for the second layer.
            w3 (ColumnParallelLinear): Linear transformation for the third layer.

        """
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MLP(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.c_fc = nn.Linear(args.n_embd, args.widen_factor * args.n_embd, bias=False)
        self.c_fc.g_scale = args.n_embd**(-1)
        if args.act_name.lower() == 'gelu':
            self.act = nn.GELU(approximate='tanh')
        elif args.act_name.lower() == 'relu':
            self.act = nn.ReLU()
        self.c_proj = nn.Linear(args.widen_factor * args.n_embd, args.n_embd, bias=False)
        self.c_proj.g_scale = (args.widen_factor * args.n_embd)**(-1)
        self.dropout = nn.Dropout(args.dp)
        return

    def forward(self, x: Tensor) -> Tensor:
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x