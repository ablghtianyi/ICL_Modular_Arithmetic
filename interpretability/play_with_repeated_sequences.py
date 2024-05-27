import os
import sys
paths_to_add = ["..", "../.."]
for path in paths_to_add:
    sys_path = os.path.relpath(path)
    if sys_path not in sys.path:  # Check to avoid duplicates
        sys.path.append(sys_path)

import argparse
import random
import torch

from tqdm import tqdm
import pickle

from _src.datasets import prepare_data_grid
from _src.models import RoPETransformer, RoPEFlashAttention
from _src.eval_utils import measure_grid_accloss_new
from _src.task_utils import str2bool, generate_all_unique_sublists, generate_all_unique_sublists_givenWs, parallelogram_tasks_with_shared_components, get_ood_lists


import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.size": 24})


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
parser.add_argument("--path_to_results", default="./", type=str, help="Path to save results")

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
args.n_point_per_row = 2

device = torch.device(args.device)
if 'cuda' in args.device:
    torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True)

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


with torch.no_grad():
    ## Generate in-distriution tasks

    pl_Ws = generate_all_unique_sublists(args, n_tasks=args.n_tasks_pl)
    if args.parallelogram is True:
        pl_Ws = parallelogram_tasks_with_shared_components(pl_Ws, args)
    rd_Ws = generate_all_unique_sublists_givenWs(pl_Ws, args, n_tasks=args.n_tasks_rd)
    pre_Ws = pl_Ws + rd_Ws
    args.n_tasks = len(pre_Ws)
    args.pre_train_n_tasks = len(pre_Ws)
    pre_Ws = [tuple(W) for W in pre_Ws]
        
    grid_set_pre, tokenizer_pre = prepare_data_grid(args, pre_Ws)

    """Make copies of data"""
    task_rows_pre = args.n_tasks
    grid_set_pre = grid_set_pre.view(task_rows_pre, len(grid_set_pre) // task_rows_pre, -1)


    ## Generate out-of-distibution tasks

    ood_Ws = get_ood_lists(pre_Ws, args)
    ood_Ws = [tuple(W) for W in ood_Ws]
    args.n_ood_tasks = len(ood_Ws)

    grid_set_ood, tokenizer_ood = prepare_data_grid(args, ood_Ws)

    """Make copies of data"""
    task_rows_ood = args.n_ood_tasks
    grid_set_ood = grid_set_ood.view(task_rows_ood, len(grid_set_ood) // task_rows_ood, -1)


    # Ws = list(itertools.product(range(1, args.p), repeat=args.n_var))
    # args.n_tasks = len(Ws)
    # print('All Ws: \n', Ws, args.n_tasks)


    ## set some args
    args.vocab_size = tokenizer_pre.__len__()
    args.max_digits = tokenizer_pre.max_digits
    args.max_digits = 2 * args.max_digits if args.pos_hint is True else args.max_digits
    args.dim = args.max_digits * (len(pre_Ws[0]) + 1)

        
    ## Load model

    if args.n_tasks_rd == 0:
        ckpt_path = f'../ckpts/heal/heal_set{args.mask_set_id}_upto{args.layer_id}_noembd{args.dont_decay_embd}_parale{args.parallelogram}_{args.model_name}_p{args.p}_base{args.base}_row32_ntask{args.pre_train_n_tasks}_nvar{args.n_var}_dsplit{args.split_data}_dfrac{args.data_pct:.1f}_{args.act_name}_n{args.n_embd}_h{args.n_head}_d{args.n_layer}_lctx{args.block_size}_I{args.seed}_dI{args.data_seed}_{args.optim}_bs{args.bs}_Tf{args.tune_steps}_T{args.steps:d}_Tw{args.warmup_steps:d}_Trshf{args.reshuffle_step}_lr{args.lr:0.2e}_wd{args.wd:.2e}.pth'
    else:
        raise ValueError("not implemented")
    
    ## 2-layer checkpoint
    # ckpt_path = "../ckpts/heal/noembdFalse_paraleTrue_pltask512_rope_decoder_p29_base29_row32_ntask512_nvar2_dsplitTrue_dfrac60.0_relu_n512_h4_d2_lctx512_I2_dI0_adamw_bs1024_t200000_T200000_Tw10000_lr1.50e-04_wd2.00e+00.pth"
    
    ## 6-layer checkpoint
    ckpt_path = "../ckpts/heal/noembdFalse_paraleTrue_pltask512_rope_decoder_p29_base29_row32_ntask512_nvar2_dsplitTrue_dfrac60.0_relu_n512_h4_d6_lctx512_I2_dI0_adamw_bs1024_t200000_T200000_Tw10000_lr1.50e-04_wd2.00e+00.pth"

    
    args.n_layer = 2

    if args.model_name == 'rope_decoder':
        assert args.s == 0.0
        if args.layer_id > args.n_layer:
            args.layer_id = args.n_layer
    model = RoPETransformer(RoPEFlashAttention, args).to(device=device)
    
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu'), strict=False)
    model.to(device)
    model.eval()


    ## Eval loop for pre_Ws

    args.eval_bs_pre = len(pre_Ws)
    args.eval_bs = args.eval_bs_pre    

    ids = []
    accs_pre = []
    losses_pre = []
    logits_pre = []
    preds_pre = []
    accs_lastshot_pre = []
    losses_lastshot_pre = []
    logits_lastshot_pre = []
    preds_lastshot_pre = []

    ## set the seed for this loop
    random.seed(args.seed - 1)

    if args.tqdm_bar:
        iterator = tqdm(range(1, args.total_shots+1))
    else:
        iterator = range(1, args.total_shots+1)

    for shot in iterator:
        
        id = random.randint(0, grid_set_pre.shape[1]-1)
        ids.append(id)
        assert len(ids) == len(set(ids))
        
        example = torch.stack([grid_set_pre[:, id, : ]] * grid_set_pre.shape[1], dim=1 )
        if shot == 1:
            previous_examples = example
        else:
            previous_examples = torch.cat([previous_examples, example], dim=2)
        
        ## grid for the last shot
        grid_lastshot = torch.cat( [torch.zeros_like(grid_set_pre)]*2, dim=2 )
        grid_lastshot[:, :, :args.dim] = example
        grid_lastshot[:, :, args.dim:] = grid_set_pre
        
        ## grid for k-shot
        grid = torch.cat( [torch.zeros_like(grid_set_pre)] * (shot+1), dim=2 )
        if shot != 1:
            grid[:, :, : shot * args.dim] = previous_examples
        grid[:, :, (shot-1) * args.dim : shot * args.dim] = example
        grid[:, :, shot * args.dim :] = grid_set_pre
        
        ## evaluate last shot
        acc_lastshot, loss_lastshot, logit_lastshot, pred_lastshot = measure_grid_accloss_new(model, grid_lastshot, args, device, n_measure=args.n_measure)
        
        ## reshape evaluation data
        # acc_lastshot = acc_lastshot.reshape((args.eval_bs, args.p, args.p, args.n_point_per_row))
        # loss_lastshot = loss_lastshot.reshape((args.eval_bs, args.p, args.p, args.n_point_per_row))
        # logit_lastshot = logit_lastshot.reshape((args.eval_bs, args.p, args.p, args.n_point_per_row, args.p))
        # pred_lastshot = pred_lastshot.reshape((args.eval_bs, args.p, args.p, args.n_point_per_row))
        acc_lastshot = acc_lastshot.reshape((args.eval_bs, args.p, args.p))
        loss_lastshot = loss_lastshot.reshape((args.eval_bs, args.p, args.p))
        logit_lastshot = logit_lastshot.reshape((args.eval_bs, args.p, args.p, args.p))
        pred_lastshot = pred_lastshot.reshape((args.eval_bs, args.p, args.p))
        
        ## append evaluation data
        accs_lastshot_pre.append(acc_lastshot[:, :, :])
        losses_lastshot_pre.append(loss_lastshot[:, :, :])
        logits_lastshot_pre.append(logit_lastshot[:, :, :, :])
        preds_lastshot_pre.append(pred_lastshot[:, :, :])
        
        ## change args.n_point_per_row according to the shot
        args.n_point_per_row = shot+1
        
        ## evaluate k-shot
        acc, loss, logit, pred = measure_grid_accloss_new(model, grid, args, device, n_measure=args.n_measure)
        
        ## reshape evaluation data
        acc = acc.reshape((args.eval_bs, args.p, args.p))
        loss = loss.reshape((args.eval_bs, args.p, args.p))
        logit = logit.reshape((args.eval_bs, args.p, args.p, args.p))
        pred = pred.reshape((args.eval_bs, args.p, args.p))
        
        ## append evaluation data
        accs_pre.append(acc[:, :, :])
        losses_pre.append(loss[:, :, :])
        logits_pre.append(logit[:, :, :, :])
        preds_pre.append(pred[:, :, :])
        
        ## change args.n_point_per_row back to 2
        args.n_point_per_row = 2
        
        
    ## Eval loop for ood_Ws
        
    args.eval_bs_ood = len(ood_Ws)
    args.eval_bs = args.eval_bs_ood

    accs_ood = []
    losses_ood = []
    logits_ood = []
    preds_ood = []
    accs_lastshot_ood = []
    losses_lastshot_ood = []
    logits_lastshot_ood = []
    preds_lastshot_ood = []
    
    random.seed(args.seed)


    for shot in tqdm(range(1, args.total_shots+1)):
        
        ## use the same ids for ood tasks as pre tasks
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
        
        ## evaluate last shot
        acc_lastshot, loss_lastshot, logit_lastshot, pred_lastshot = measure_grid_accloss_new(model, grid_lastshot, args, device, n_measure=args.n_measure)
        
        ## reshape evaluation data
        acc_lastshot = acc_lastshot.reshape((args.eval_bs, args.p, args.p))
        loss_lastshot = loss_lastshot.reshape((args.eval_bs, args.p, args.p))
        logit_lastshot = logit_lastshot.reshape((args.eval_bs, args.p, args.p, args.p))
        pred_lastshot = pred_lastshot.reshape((args.eval_bs, args.p, args.p))
        
        ## append evaluation data
        accs_lastshot_ood.append(acc_lastshot[:, :, :])
        losses_lastshot_ood.append(loss_lastshot[:, :, :])
        logits_lastshot_ood.append(logit_lastshot[:, :, :, :])
        preds_lastshot_ood.append(pred_lastshot[:, :, :])
        
        ## change args.n_point_per_row according to the shot
        args.n_point_per_row = shot+1
        
        ## evaluate k-shot
        acc, loss, logit, pred = measure_grid_accloss_new(model, grid, args, device, n_measure=args.n_measure)
        
        ## reshape evaluation data
        acc = acc.reshape((args.eval_bs, args.p, args.p))
        loss = loss.reshape((args.eval_bs, args.p, args.p))
        logit = logit.reshape((args.eval_bs, args.p, args.p, args.p))
        pred = pred.reshape((args.eval_bs, args.p, args.p))
        
        ## append evaluation data
        accs_ood.append(acc[:, :, :])
        losses_ood.append(loss[:, :, :])
        logits_ood.append(logit[:, :, :, :])
        preds_ood.append(pred[:, :, :])
        
        ## change args.n_point_per_row back to 2
        args.n_point_per_row = 2


data = {}
data['pre_Ws'] = pre_Ws
data['ood_Ws'] = ood_Ws
data['grid_set_pre'] = grid_set_pre
data['grid_set_ood'] = grid_set_ood
data['ids'] = ids
data['accs_pre'] = accs_pre
data['losses_pre'] = losses_pre
data['logits_pre'] = logits_pre
data['preds_pre'] = preds_pre
data['accs_lastshot_pre'] = accs_lastshot_pre
data['losses_lastshot_pre'] = losses_lastshot_pre
data['logits_lastshot_pre'] = logits_lastshot_pre
data['preds_lastshot_pre'] = preds_lastshot_pre
data['accs_ood'] = accs_ood
data['losses_ood'] = losses_ood
data['logits_ood'] = logits_ood
data['preds_ood'] = preds_ood
data['accs_lastshot_ood'] = accs_lastshot_ood
data['losses_lastshot_ood'] = losses_lastshot_ood
data['logits_lastshot_ood'] = logits_lastshot_ood
data['preds_lastshot_ood'] = preds_lastshot_ood

filename = f"grid_{args.n_layer}layer_{args.total_shots}-shot_set{args.mask_set_id}.pickle"
with open(args.path_to_results + filename, 'wb') as f:
    pickle.dump(data, f)