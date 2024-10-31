import os
import sys
paths_to_add = ["..", "../.."]
for path in paths_to_add:
    sys_path = os.path.relpath(path)
    if sys_path not in sys.path:
        sys.path.append(sys_path)

import math
import argparse
import itertools
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from _src.datasets import prepare_data
from _src.models import RoPETransformer, RoPEFlashAttention
from _src.task_utils import str2bool, generate_all_unique_sublists, generate_all_unique_sublists_givenWs, parallelogram_tasks_with_shared_components, get_ood_lists

from matplotlib import colors as mcolors
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({"font.size": 24})
sns.set_theme(style="darkgrid")
cmap0 = sns.diverging_palette(250, 30, l=65, center="dark", as_cmap=True)

parser = argparse.ArgumentParser(description="Transformer ICL")
# Basic setting
parser.add_argument("--model_name", default="rope_decoder", type=str, help="Encoder or Decoder only Transformers")
parser.add_argument("--device", default="cuda", type=str, help="device")
parser.add_argument("--dtype", default="float32", type=str, help="dtype")
parser.add_argument("--mixed_precision", default=False, type=str2bool, help="Automatic Mixed Precision")
parser.add_argument("--num_workers", default=0, type=int, help="Workers for Datalodaer")
parser.add_argument("--seed", default=0, type=int, help="random seed")

# Model Settings
parser.add_argument("--n_layer", default=2, type=int, help="Number of Transformer Blocks")
parser.add_argument("--dp", default=0.0, type=float, help="Dropout Probability")
parser.add_argument("--if_ln", default=True, type=str2bool, help="If use LayerNorm or Not")
parser.add_argument("--vocab_size", default=-1, type=int, help="Vocab size, later updated while generating dataset")
parser.add_argument("--n_embd", default=256, type=int, help="Embedding Dimension")
parser.add_argument("--n_head", default=4, type=int, help="Number of Heads")
parser.add_argument("--block_size", default=256, type=int, help='maximum ctx length')
parser.add_argument("--act_name", default="relu", type=str, help="activation: relu, gelu, swiglu")
parser.add_argument("--widen_factor", default=4, type=int, help="MLP widening")
parser.add_argument("--mu", default=1.0, type=float, help="Skip connection strength")
parser.add_argument("--weight_tying", default=False, type=str2bool, help="If use weight tying")
parser.add_argument("--dont_decay_embd", default=True, type=str2bool, help="True means weight decay is not applied to Embedding layer")
parser.add_argument("--s", default=0.0, type=float, help="s=0 for SP, 1 for muP like attention. Use 0.0 only for now.")

# Data
parser.add_argument("--n_tasks_pl", default=4, type=int, help="number of independent tasks")
parser.add_argument("--n_tasks_rd", default=4, type=int, help="number of independent tasks")
parser.add_argument("--parallelogram", default=True, type=str2bool, help="Perform parallelogram construction on task vectors or not, n_tasks will be multiplied by 4 later")
parser.add_argument("--n_var", default=2, type=int, help="number of variables, x, y, z, ...")
parser.add_argument("--data_seed", default=42, type=int, help="random seed for generating datasets")
parser.add_argument("--split_data", default=True, type=str2bool, help="Train/Test set have different inputs or not.")
parser.add_argument("--data_pct", default=50.0, type=float, help="Data Percentage")
parser.add_argument("--split_tasks", default=False, type=str2bool, help="Set to False, ood test is done seperately now")
parser.add_argument("--task_pct", default=50.0, type=float, help="Task Percentage, not used if split_tasks is set to False")
parser.add_argument("--p", default=97, type=int, help="Modulo p")
parser.add_argument("--base", default=97, type=int, help="Represent Numbers in base")
parser.add_argument("--n_point_per_row", default=16, type=int, help="Number of examples (x, y, ..., f(x, y, ...)) per row")
parser.add_argument("--encrypted", default=True, type=str2bool, help="Write the task vectors in seq or not. Keep True for ICL.")
parser.add_argument("--pos_hint", default=False, type=str2bool, help="Add positional hint for each digit or not")
parser.add_argument("--reverse_target", default=False, type=str2bool, help="Reverse the digits order of targets or not")
parser.add_argument("--show_mod", default=False, type=str2bool, help="Add <mod> <p> to token or not")
parser.add_argument("--show_seos", default=False, type=str2bool, help="Use <SOS> and <EOS> or not")
parser.add_argument("--n_val_step", default=10, type=int, help="How many batches per evaluation")

# Optimization
parser.add_argument("--optim", default="adamw", type=str, help="optimizer, use adamw only")
parser.add_argument("--bs", default=1024, type=int, help="Batchsize for training")
parser.add_argument("--eval_bs", default=512, type=int, help="Batchsize for evaluation")
parser.add_argument("--lr", default=5e-4, type=float, help="Learning Rate. We fix warmup initial lr to be 0.01 * lr, final lr to be 0.1 * lr")
parser.add_argument("--n_cycles", default=1, type=int, help="Cycles of scheduler, only use 1 cycle.")
parser.add_argument("--clip", default=0.0, type=float, help="Gradient clip, 0.0 means not used.")
parser.add_argument("--wd", default=0.5, type=float, help="Weight decay")
parser.add_argument("--beta1", default=0.9, type=float, help="Beta 1 for AdamW")
parser.add_argument("--beta2", default=0.98, type=float, help="Beta 2 for AdamW")
parser.add_argument("--eps", default=1e-8, type=float, help="Eps for AdamW")
parser.add_argument("--steps", default=200000, type=int, help="Training steps")
parser.add_argument("--warmup_steps", default=10000, type=int, help="Warmup steps")
parser.add_argument("--lr_decay", default='cosine', type=str, help="If Use Scheduler, cosine and linear are allowed")
parser.add_argument("--reshuffle_step", default=1, type=int, help="Keep to 1, old flag, no longer used.")
parser.add_argument("--fake_epochs", default=1000000, type=int, help="fake epochs, keep any large number")

# Saving
parser.add_argument("--savefig", default=False, type=str2bool, help="Save Fig or not")
parser.add_argument("--n_measure", default=1, type=int, help="Number of batches for evaluations")
parser.add_argument("--end_pos", default=3, type=int, help="The position of the token along the sequence. Only (end_pos // 3) matters.")
parser.add_argument("--task_id", default=0, type=int, help="Task_id, 0-(p-1)**2, (1, 1), (1, 2), ...")
# parser.add_argument("--plot_layer_idx", default=0, type=int, help="Which head to plot, not always used")
# parser.add_argument("--plot_head_idx", default=0, type=int, help="Which head to plot, not always used")


# Colormap
N = 5
base_cmaps = ['Greys', 'Purples', 'Reds', 'Blues', 'Oranges', 'Greens']
n_base = len(base_cmaps)
colors = np.concatenate([plt.get_cmap(name)(np.linspace(0.5, 1.0, N)) for name in base_cmaps])
custom_cmap = mcolors.ListedColormap(colors)


@torch.inference_mode()
def scaled_dot_product_attention(query, key, value=None, attn_mask=None, dropout_p=0.0, is_causal=True, scale=None) -> torch.Tensor:
    # Efficient implementation equivalent to the following:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    if value is not None:
        attn_weight = torch.dropout(attn_weight, dropout_p, train=False)
        output = (attn_weight @ value).cpu()
    else:
        output = None
    
    return attn_weight.cpu(), output


@torch.inference_mode()
def svd_flip(u, v):
    # columns of u, rows of v
    max_abs_cols = torch.argmax(torch.abs(u), 0)
    idx = torch.arange(u.shape[1]).to(u.device)
    signs = torch.sign(u[max_abs_cols, idx])
    u *= signs
    v *= signs.view(-1, 1)
    return u, v


@torch.inference_mode()
def pca(embedding, n_top_direction = 2):
    n_embd, vocab_size = embedding.shape
    
    mean = embedding.mean(dim=1, keepdim=True) # (n_embd, 1)
    centered_data = embedding - mean

    U, S, Vt = torch.linalg.svd(centered_data.T, full_matrices=False)
    top_direction = Vt[:n_top_direction, :]  # (n_top_direction, n_embd)
    
    U, Vt = svd_flip(U, Vt)
    
    return (top_direction + mean.T)


def find_log(base, target, p):
    for x in range(1, p):
        if pow(base, x, p) == target:
            return x
    return 0


# Main funtion
def main(args):

    # Init distributed mode
    device = torch.device(args.device)
    if "cuda" in args.device:
        args.ngpu_per_node = torch.cuda.device_count()
    else:
        args.ngpu_per_node = 1

    # Check args
    assert args.encrypted == True
    assert args.show_seos == False
    assert args.pos_hint == False
    assert args.reverse_target == False
    assert args.show_mod == False
    if args.weight_tying == True:
        assert args.dont_decay_embd == False
    else:
        assert args.dont_decay_embd == True
    
    if args.mixed_precision is True:
        assert args.dtype in ['float16', 'bfloat16']
        assert 'cuda' in args.device
        if args.dtype == 'float16':
            args.dtype = torch.float16
        else:
            args.dtype = torch.bfloat16
    else:
        torch.set_float32_matmul_precision('high')
        args.dtype = torch.float32

    # Setup seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Generate Task vectors
    pl_Ws = generate_all_unique_sublists(args, n_tasks=args.n_tasks_pl)
    if args.parallelogram is True:
        pl_Ws = parallelogram_tasks_with_shared_components(pl_Ws, args)
        args.n_tasks_pl = len(pl_Ws)
    rd_Ws = generate_all_unique_sublists_givenWs(pl_Ws, args, n_tasks=args.n_tasks_rd)
    pre_Ws = pl_Ws + rd_Ws

    args.n_tasks = len(pre_Ws)
    args.max_ood_tasks = max(args.n_tasks, 512)
    ood_Ws = get_ood_lists(pre_Ws, args)

    if len(ood_Ws) > args.eval_bs:
        ood_Ws = ood_Ws[:args.eval_bs]
        
    if len(ood_Ws) % 8 != 0:
        args.n_ood_tasks = (len(ood_Ws) // 8) * 8
        ood_Ws = ood_Ws[:args.n_ood_tasks]
        args.ood_bs = len(ood_Ws) * args.ngpu_per_node
    else:
        args.n_ood_tasks = len(ood_Ws)
        args.ood_bs = args.n_ood_tasks * args.ngpu_per_node

    if args.n_tasks == 1:
        assert args.split_tasks == False

    # Make data
    # _, _, tokenizer = prepare_data(args, pre_Ws)
    _, _, tokenizer = prepare_data(args, ood_Ws)

    args.vocab_size = tokenizer.__len__()
    print(f'vocab size: {args.vocab_size}')
    args.max_digits = tokenizer.max_digits
    args.max_digits = 2 * args.max_digits if args.pos_hint is True else args.max_digits
    args.dim = args.max_digits * (len(pre_Ws[0]) + 1)

    # Prepare Model
    assert args.s == 0.0
    model = RoPETransformer(RoPEFlashAttention, args).to(device=device)
    
    ckpt_path = f"../ckpts/noembd{args.dont_decay_embd}_parale{args.parallelogram}_pltask{args.n_tasks_pl}_{args.model_name}_p{args.p}_base{args.base}_row{args.n_point_per_row}_ntask{args.n_tasks}_nvar{args.n_var}_dsplit{args.split_data}_dfrac{args.data_pct:.1f}_{args.act_name}_n{args.n_embd}_h{args.n_head}_d{args.n_layer}_lctx{args.block_size}_I{args.seed}_dI{args.data_seed}_{args.optim}_bs{args.bs}_t{args.steps:d}_T{args.steps:d}_Tw{args.warmup_steps:d}_lr{args.lr:0.2e}_wd{args.wd:.2e}.pth"
    
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    
    wte = model.transformer.wte.weight.data[:args.p, :].cpu()
    wte /= wte.norm(dim=-1, keepdim=True)

    # fig, ax = plt.subplots(1, 1, figsize=(6, 6), constrained_layout=True)
    # cmap = sns.color_palette('crest', n_colors=args.p)
            

    Ws = list(itertools.product(range(1, args.p), repeat=args.n_var))
    xs = np.asarray(list(itertools.product(range(0, args.p), repeat=len(Ws[0]))))
    basic_val_set = np.asarray([[(x @ W) % args.p for x in xs] for W in Ws])
    basic_val_set = np.concatenate([np.repeat(xs[None, :, :], repeats=len(Ws), axis=0), basic_val_set[:, :, None]], axis=-1) # (n_task, p^2, dim)
    print(Ws[args.task_id])
    
    rng = np.random.RandomState(args.seed + 1)
    val_set = basic_val_set
    perms = np.arange(0, args.p**2, 1)[:, None]
    for _ in range(args.n_point_per_row - 1):
        perm = rng.permutation(args.p**2)
        val_set = np.concatenate([basic_val_set[:, perm, :], val_set], axis=-1)
        inverse_perm = np.argsort(perm)[:, None]
        perms = np.concatenate([inverse_perm, perms], axis=-1)

    end_pair_id = args.end_pos // args.dim
    
    x = torch.tensor(val_set[args.task_id])
    
    for i in range(args.n_point_per_row):
        # Control the rest of sequences to be the same
        if i != end_pair_id:
            special_xs = np.random.randint(low=0, high=args.p, size=(1, 2))
            fixed_set = np.asarray([[(special_x @ W) % args.p for special_x in special_xs] for W in Ws])
            fixed_set = np.concatenate([np.repeat(special_xs[None, :, :], repeats=len(Ws), axis=0), fixed_set[:, :, None]], axis=-1) # (n_task, p^2, dim)
            x[:, i*args.dim:(i + 1)*args.dim] = torch.tensor(fixed_set[args.task_id])
    
    y = x.clone()
    
    x = x.to(device)
    y = y.to(device)
    x = x[:, :-1].contiguous()
    y = y[:, 1:].contiguous()
    
    # print(x[:3])
    logits, (qkv_list, input_list, output_list) = model.record(x)
    head_size = args.n_embd // args.n_head

    for layer_idx, (q, k, v) in enumerate(qkv_list):
        
        T = x.size(-1)
        attn_weight, head_out = scaled_dot_product_attention(q, k, value=v)

        for head_idx in range(args.n_head):
            fig_attn, ax_attn = plt.subplots(1, 1, figsize=(6, 6), constrained_layout=True)
            
            attn_weight_plot = attn_weight[0, head_idx, :, :].cpu()
            attn_weight_plot[attn_weight_plot == 0.0] = torch.nan
            sns.heatmap(attn_weight_plot.numpy()[:48, :48], annot=False, cmap='coolwarm', fmt='.2f', ax=ax_attn, vmin=0, vmax=1, 
                        annot_kws={"ha": 'center', "va": 'center'}, )
            
            ax_attn.tick_params(axis='both', which='major', length=5, width=0.5, reset=True)
            ax_attn.tick_params(bottom=True, top=False, left=True, right=False)
            
            tick_positions = np.arange(-1, 48, args.dim)
            tick_positions[0] = 0
            # Set ticks and labels
            ax_attn.set_xticks(tick_positions + 0.5)
            ax_attn.set_yticks(tick_positions + 0.5)
            ax_attn.set_xticklabels(tick_positions, fontsize=12, rotation=0, va='center')
            ax_attn.set_yticklabels(tick_positions, fontsize=12, rotation=45, ha='center')
            
            if args.savefig is True:
                fig_path = f'./attn_map/layer{layer_idx}_head{head_idx}_Ws{Ws[args.task_id][0]}_{Ws[args.task_id][1]}_{args.model_name}_p{args.p}_base{args.base}_row{args.n_point_per_row}_task{Ws[args.task_id]}_ntask{args.n_tasks}_nvar{args.n_var}_dsplit{args.split_data}_dfrac{args.data_pct:.1f}_{args.act_name}_n{args.n_embd}_h{args.n_head}_d{args.n_layer}_lctx{args.block_size}_I{args.seed}_dI{args.data_seed}_{args.optim}_bs{args.bs}_T{args.steps:d}_Tw{args.warmup_steps:d}_Trshf{args.reshuffle_step}_lr{args.lr:0.2e}_wd{args.wd:.2e}.pdf'
                fig_attn.savefig(fig_path, format='pdf')
            else:
                plt.show()
            
            plt.close(fig_attn)

        
        
    return

if __name__ == '__main__':
    args = parser.parse_args()
    with torch.inference_mode():
        main(args)
