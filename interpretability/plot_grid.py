import os
import sys
paths_to_add = ["..", "../.."]
for path in paths_to_add:
    sys_path = os.path.relpath(path)
    if sys_path not in sys.path:  # Check to avoid duplicates
        sys.path.append(sys_path)

import argparse
import numpy as np
import torch
from tqdm import tqdm
import pickle

from _src.task_utils import str2bool

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.size": 16})

def find_pairs_matrix_multiple(pairs, p=29):
    # Initialize a 2D array with zeros
    matrix = np.zeros((p, p), dtype=int)
    
    for pair in pairs:
        x1, y1 = pair
        for c in range(p):
            x = (c * x1) % p
            y = (c * y1) % p
            matrix[x, y] = 1  # Set the corresponding position to 1
    
    return matrix


parser = argparse.ArgumentParser(description="Transformer ICL")
# Basic setting
parser.add_argument("--model_name", default="rope_decoder", type=str, help="Encoder or Decoder only Transformers")
parser.add_argument("--device", default="cpu", type=str, help="device")
parser.add_argument("--dtype", default="float32", type=str, help="dtype")
parser.add_argument("--mixed_precision", default=False, type=str2bool, help="Automatic Mixed Precision")
parser.add_argument("--seed", default=1, type=int, help="random seed")
parser.add_argument("--num_workers", default=0, type=int, help="Workers for Datalodaer")
parser.add_argument("--world_size", default=1, type=int, help="World Size")
parser.add_argument("--ddp", default=False, type=str2bool, help="DDP mode")
parser.add_argument("--tqdm_bar", default=False, type=str2bool, help="Enable tqdm bar or not")

# ddp Settings
parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
parser.add_argument('--dist_backend', default='nccl', help='gloo or nccl ...')

# Model Settings
parser.add_argument("--n_layer", default=4, type=int, help="Number of Transformer Blocks")
parser.add_argument("--dp", default=0.0, type=float, help="Dropout Probability")
parser.add_argument("--if_ln", default=True, type=str2bool, help="If use LayerNorm or Not")
parser.add_argument("--n_embd", default=512, type=int, help="Embedding Dimension")
parser.add_argument("--n_head", default=4, type=int, help="Number of Heads")
parser.add_argument("--block_size", default=512, type=int, help='maximum length')
parser.add_argument("--act_name", default="relu", type=str, help="activation: relu, gelu, swiglu")

parser.add_argument("--n_question_per_row", default=16, type=int, help="Content length, -1 means depends on data.")
parser.add_argument("--widen_factor", default=4, type=int, help="MLP widening")
parser.add_argument("--mu", default=1.0, type=float, help="Skip connection strength")
parser.add_argument("--weight_tying", default=False, type=str2bool, help="If use weight tying")
parser.add_argument("--dont_decay_embd", default=False, type=str2bool, help="If use weight tying")
parser.add_argument("--s", default=0.0, type=float, help="s=0 for SP, 1 for muP like attention. Use 0.0 only for now.")

# Data
parser.add_argument("--n_tasks_pl", default=96, type=int, help="number of parallelogram tasks")
parser.add_argument("--n_tasks_rd", default=0, type=int, help="number of new random tasks")
parser.add_argument("--parallelogram", default=True, type=str2bool, help="Perform parallelogram construction on task vectors or not")
parser.add_argument("--n_var", default=2, type=int, help="number of variables, i.e. dimension of the problem")
parser.add_argument("--data_seed", default=0, type=int, help="random seed for generating datasets")
parser.add_argument("--data_pct", default=80.0, type=float, help="Data Percentage")
parser.add_argument("--task_pct", default=50.0, type=float, help="Task Percentage")
parser.add_argument("--p", default=29, type=int, help="Modulo p")
parser.add_argument("--base", default=29, type=int, help="Represent Numbers in base")
parser.add_argument("--n_point_per_row", default=2, type=int, help="Number of data points per row")
parser.add_argument("--plot_ctx", default=2, type=int, help="k-shot")
parser.add_argument("--end_pos", default=-1, type=int, help="k")
parser.add_argument("--start_pos", default=0, type=int, help="k")
parser.add_argument("--encrypted", default=True, type=str2bool, help="Write the task vectors in data or not.")
parser.add_argument("--pos_hint", default=False, type=str2bool, help="Add positional hint or not")
parser.add_argument("--reverse_target", default=False, type=str2bool, help="Reverse the digits order of targets or not")
parser.add_argument("--show_mod", default=False, type=str2bool, help="Add mod p to token or not")
parser.add_argument("--show_seos", default=False, type=str2bool, help="USe SOS and EOS or not")
parser.add_argument("--split_tasks", default=False, type=str2bool, help="Train/Test set have different task vectors or not.")
parser.add_argument("--split_data", default=True, type=str2bool, help="Train/Test set have different datapoints or not.")
parser.add_argument("--fake_restart_steps", default=5000, type=int, help="Fake restart steps, to save memory")

# Optimization
parser.add_argument("--optim", default="adamw", type=str, help="Optimizer: adamw or sgd")
parser.add_argument("--bs", default=1536, type=int, help="Batchsize")
parser.add_argument("--eval_bs", default=1, type=int, help="Batchsize for Evaluation")
parser.add_argument("--lr", default=1.50e-4, type=float, help="Learning Rate")
parser.add_argument("--n_cycles", default=1, type=int, help="Cycles of scheduler, only use 1 cycle.")
parser.add_argument("--clip", default=0.0, type=float, help="Gradient clip, 0.0 means not used.")
parser.add_argument("--wd", default=2.0, type=float, help="Weight Decay")
parser.add_argument("--beta1", default=0.9, type=float, help="Beta 1 for AdamW")
parser.add_argument("--beta2", default=0.98, type=float, help="Beta 2 for AdamW")
parser.add_argument("--eps", default=1e-8, type=float, help="Eps for AdamW")
parser.add_argument("--steps", default=200000, type=int, help="Training steps")
parser.add_argument("--warmup_steps", default=10000, type=int, help="Warmup steps")
parser.add_argument("--tune_steps", default=10000, type=int, help="Training steps")
parser.add_argument("--tune_warmup_steps", default=1000, type=int, help="Warmup steps")
parser.add_argument("--lr_decay", default='cosine', type=str, help="If Use Scheduler")
parser.add_argument("--steps_per_record", default=200, type=int, help="Save Results")
parser.add_argument("--reshuffle_step", default=1, type=int, help="Save Results")
parser.add_argument("--fake_epochs", default=1000000, type=int, help="fake epochs, keep any large number")

# Inference
parser.add_argument("--prune_mode", default='zero', type=str, help="How to prune attention heads")
parser.add_argument("--layer_id", default=4, type=int, help="Up to which layer we extract information")
parser.add_argument("--n_prune_heads", default=-1, type=int, help="How many heads to prune")
parser.add_argument("--mask_set_id", default=0, type=int, help="Special set to mask")
parser.add_argument("--freeze_set_id", default=-1, type=int, help="Freeze Key Heads or Not")
parser.add_argument("--new_head", default='none', type=str, help="Reinitialize lm head or not")
parser.add_argument("--train_set", default=False, type=str2bool, help="Use training inputs or val inputs")
parser.add_argument("--n_measure", default=8, type=int, help="How many batches to average.")
parser.add_argument("--g_seed", default=2, type=int, help="random generator seed")
parser.add_argument("--savefig", default=False, type=str2bool, help="Save Figure")
parser.add_argument("--total_shots", default=2, type=int, help="number of in-context examples")

args = parser.parse_args()
assert args.show_seos == False
assert args.split_tasks == False
args.base = args.p

colors = [(1, 0, 0), (1, 1, 1), (0, 0, 1)]  # Red, White, Blue
n_bins = 3  # Discretize into 3 bins
cmap_name = 'red_white_blue'
cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)


with open(f'../data/sequences/grid_2layer_{args.total_shots}-shot_set{args.mask_set_id}.pickle', 'rb') as f:
    data = pickle.load(f)

pre_Ws = data['pre_Ws']
grid_set_pre = data['grid_set_pre']
ids = data['ids']
accs_pre = data['accs_pre']
accs_lastshot_pre = data['accs_lastshot_pre']

accs_rescale_pre = []
accs_sum_pre = []
pairs = []
for shot in range(1, args.total_shots+1):
    id = ids[shot-1]
    accs_pre[shot-1] = accs_pre[shot-1].astype(int)
    accs_lastshot_pre[shot-1] = accs_lastshot_pre[shot-1].astype(int)
    
    pairs.append(grid_set_pre[0, id, :2])
    accs_rescale_pre.append(find_pairs_matrix_multiple(pairs))
    if shot == 1:
        accs_sum_pre.append(accs_rescale_pre[-1])
    else:
        accs_sum_pre.append(np.logical_or(accs_rescale_pre[-1], accs_sum_pre[-1]))

args.max_digits = 1
args.max_digits = 2 * args.max_digits if args.pos_hint is True else args.max_digits
args.dim = args.max_digits * (len(pre_Ws[0]) + 1)

for i in tqdm(range(accs_pre[0].shape[0])):

    nrows = 6
    plt.figure(figsize=(args.total_shots*5, nrows*5))
    plt.suptitle(f"task: {pre_Ws[i]}")

    for shot in range(1, args.total_shots+1):
        id = ids[shot-1]
        
        example = torch.stack([grid_set_pre[:, id, : ]] * grid_set_pre.shape[1], dim=1 )
        if shot == 1:
            previous_examples = example
        else:
            previous_examples = torch.cat([previous_examples, example], dim=2)
        
        # grid for the last shot
        grid_lastshot = torch.cat( [torch.zeros_like(grid_set_pre)]*2, dim=2 )
        grid_lastshot[:, :, :args.dim] = example
        grid_lastshot[:, :, args.dim:] = grid_set_pre
        
        # grid for k-shot
        grid = torch.cat( [torch.zeros_like(grid_set_pre)] * (shot+1), dim=2 )
        if shot != 1:
            grid[:, :, : shot * args.dim] = previous_examples
        grid[:, :, (shot-1) * args.dim : shot * args.dim] = example
        grid[:, :, shot * args.dim :] = grid_set_pre
        
        plt.subplot(nrows, args.total_shots, shot)
        # title_string = grid[i, 0, :-args.dim].numpy()
        # title_string = ' '.join(title_string.astype(str))
        # title_string = title_string.split()
        # title_string = '(' + ') ('.join(' '.join(title_string[i:i+3]) for i in range(0, len(title_string), 3)) + ')'
        # plt.title(f'{shot}-shot: {title_string} (x, y, ?)')
        plt.title(f'{shot}-shot (avg_acc={100 * accs_pre[shot-1][i, :, :].mean():.2f}%)')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.imshow(accs_pre[shot-1][i, :, :].transpose(), origin='lower')
        
        plt.subplot(nrows, args.total_shots, args.total_shots + shot)
        plt.title(f'rescale: {tuple( grid_lastshot[i, 0, -2 * args.dim : -args.dim].numpy() )} (x, y, ?)')
        plt.xlabel('x')
        plt.ylabel('y')
        # plt.imshow(accs_lastshot_pre[shot-1][i, :, :].transpose(), origin='lower')
        plt.imshow(accs_rescale_pre[shot-1].transpose(), origin='lower')
        
        plt.subplot(nrows, args.total_shots, 2*args.total_shots + shot)
        plt.title(f'add rescalings so far (avg_acc={100 * accs_sum_pre[shot-1][:, :].mean():.2f}%)')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.imshow(accs_sum_pre[shot-1][:, :].transpose(), origin='lower')
        
        plt.subplot(nrows, args.total_shots, 3*args.total_shots + shot)
        plt.title(f'{shot}-shot > rescalings so far')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.imshow(( accs_pre[shot-1][i, :, :] & ~(accs_sum_pre[shot-1][:, :]) ).transpose(), origin='lower')
        
        plt.subplot(nrows, args.total_shots, 4*args.total_shots + shot)
        plt.title(f'rescalings so far > {shot}-shot')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.imshow(( ~(accs_pre[shot-1][i, :, :]) & accs_sum_pre[shot-1][:, :] ).transpose(), origin='lower')
        
        plt.subplot(nrows, args.total_shots, 5*args.total_shots + shot)
        plt.title(f'acc {shot}-shot - acc rescalings so far \n (avg = {100 * (accs_pre[shot-1][i, :, :].mean() - accs_sum_pre[shot-1][:, :].mean()):.2f})')
        plt.xlabel('x')
        plt.ylabel('y')
        cax = plt.imshow(( accs_pre[shot-1][i, :, :] - (accs_sum_pre[shot-1][:, :]) ).transpose(), origin='lower', cmap=cm, vmin=-1, vmax=1)
        cbar = plt.colorbar(cax, ticks=[-1, 0, 1])
        cbar.ax.set_yticklabels(['-1', '0', '1'])

    # plt.subplots_adjust(wspace=-0.7, hspace=0.4)
    # plt.tight_layout()
    plt.savefig(f'../data/sequences/new_2layer_{args.total_shots}-shot/set{args.mask_set_id}/accs_pre/grid_{args.total_shots}-shot_pre_task={pre_Ws[i][0]},{pre_Ws[i][1]}.pdf', format='pdf', bbox_inches='tight')    
    
    plt.close()


ood_Ws = data['ood_Ws']
grid_set_ood = data['grid_set_ood']
ids = data['ids']
accs_ood = data['accs_ood']
accs_lastshot_ood = data['accs_lastshot_ood']

accs_rescale_ood = []
accs_sum_ood = []
for shot in range(1, args.total_shots+1):
    id = ids[shot-1]
    accs_ood[shot-1] = accs_ood[shot-1].astype(int)
    accs_lastshot_ood[shot-1] = accs_lastshot_ood[shot-1].astype(int)
    
    pairs.append(grid_set_ood[0, id, :2])
    accs_rescale_ood.append(find_pairs_matrix_multiple(pairs))
    if shot == 1:
        accs_sum_ood.append(accs_rescale_ood[-1])
    else:
        accs_sum_ood.append(np.logical_or(accs_rescale_ood[-1], accs_sum_ood[-1]))

args.max_digits = 1
args.max_digits = 2 * args.max_digits if args.pos_hint is True else args.max_digits
args.dim = args.max_digits * (len(ood_Ws[0]) + 1)

for i in tqdm(range(accs_ood[0].shape[0])):

    plt.figure(figsize=(args.total_shots*5, 3*5))
    plt.suptitle(f"task: {ood_Ws[i]}")

    for shot in range(1, args.total_shots+1):
        id = ids[shot-1]
        
        example = torch.stack([grid_set_ood[:, id, : ]] * grid_set_ood.shape[1], dim=1 )
        if shot == 1:
            previous_examples = example
        else:
            previous_examples = torch.cat([previous_examples, example], dim=2)
        
        # grid for the last shot
        grid_lastshot = torch.cat( [torch.zeros_like(grid_set_ood)]*2, dim=2 )
        grid_lastshot[:, :, :args.dim] = example
        grid_lastshot[:, :, args.dim:] = grid_set_ood
        
        # grid for k-shot
        grid = torch.cat( [torch.zeros_like(grid_set_ood)] * (shot+1), dim=2 )
        if shot != 1:
            grid[:, :, : shot * args.dim] = previous_examples
        grid[:, :, (shot-1) * args.dim : shot * args.dim] = example
        grid[:, :, shot * args.dim :] = grid_set_ood
        
        plt.subplot(nrows, args.total_shots, shot)
        # title_string = grid[i, 0, :-args.dim].numpy()
        # title_string = ' '.join(title_string.astype(str))
        # title_string = title_string.split()
        # title_string = '(' + ') ('.join(' '.join(title_string[i:i+3]) for i in range(0, len(title_string), 3)) + ')'
        # plt.title(f'{shot}-shot: {title_string} (x, y, ?)')
        plt.title(f'{shot}-shot (avg_acc={100 * accs_ood[shot-1][i, :, :].mean():.2f}%)')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.imshow(accs_ood[shot-1][i, :, :].transpose(), origin='lower')
        
        plt.subplot(nrows, args.total_shots, args.total_shots + shot)
        plt.title(f'rescale: {tuple( grid_lastshot[i, 0, -2 * args.dim : -args.dim].numpy() )} (x, y, ?)')
        plt.xlabel('x')
        plt.ylabel('y')
        # plt.imshow(accs_lastshot_pre[shot-1][i, :, :].transpose(), origin='lower')
        plt.imshow(accs_rescale_ood[shot-1].transpose(), origin='lower')
        
        plt.subplot(nrows, args.total_shots, 2*args.total_shots + shot)
        plt.title(f'add rescalings so far (avg_acc={100 * accs_sum_ood[shot-1][:, :].mean():.2f}%)')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.imshow(accs_sum_ood[shot-1][:, :].transpose(), origin='lower')
        
        plt.subplot(nrows, args.total_shots, 3*args.total_shots + shot)
        plt.title(f'{shot}-shot > rescalings so far')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.imshow(( accs_ood[shot-1][i, :, :] & ~(accs_sum_ood[shot-1][:, :]) ).transpose(), origin='lower')
        
        plt.subplot(nrows, args.total_shots, 4*args.total_shots + shot)
        plt.title(f'rescalings so far > {shot}-shot')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.imshow(( ~(accs_ood[shot-1][i, :, :]) & accs_sum_ood[shot-1][:, :] ).transpose(), origin='lower')
        
        plt.subplot(nrows, args.total_shots, 5*args.total_shots + shot)
        plt.title(f'acc {shot}-shot - acc rescalings so far \n (avg = {100 * (accs_ood[shot-1][i, :, :].mean() - accs_sum_ood[shot-1][:, :].mean()):.2f})')
        plt.xlabel('x')
        plt.ylabel('y')
        cax = plt.imshow(( accs_ood[shot-1][i, :, :] - (accs_sum_ood[shot-1][:, :]) ).transpose(), origin='lower', cmap=cm, vmin=-1, vmax=1)
        cbar = plt.colorbar(cax, ticks=[-1, 0, 1])
        cbar.ax.set_yticklabels(['-1', '0', '1'])

    # plt.subplots_adjust(wspace=-0.7, hspace=0.4)
    # plt.tight_layout()
    plt.savefig(f'../data/sequences/new_2layer_{args.total_shots}-shot/set{args.mask_set_id}/accs_ood/grid_{args.total_shots}-shot_ood_task={ood_Ws[i][0]},{ood_Ws[i][1]}.pdf', format='pdf', bbox_inches='tight')
    
    plt.close()