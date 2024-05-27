import os
import sys
paths_to_add = ["..", "../.."]
for path in paths_to_add:
    sys_path = os.path.relpath(path)
    if sys_path not in sys.path:
        sys.path.append(sys_path)

from functools import partial
import argparse
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler

from _src.datasets import prepare_data
from _src.train_utils import train_tf_icl
from _src.ddp_utils import init_distributed_mode, get_world_size, get_rank
from _src.models import RoPETransformer, RoPEFlashAttention
from _src.task_utils import str2bool, generate_all_unique_sublists, generate_all_unique_sublists_givenWs, parallelogram_tasks_with_shared_components, get_ood_lists

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


parser = argparse.ArgumentParser(description="Transformer ICL")
# Basic setting
parser.add_argument("--model_name", default="rope_decoder", type=str, help="Encoder or Decoder only Transformers")
parser.add_argument("--device", default="cuda", type=str, help="device")
parser.add_argument("--dtype", default="bfloat16", type=str, help="dtype")
parser.add_argument("--mixed_precision", default=True, type=str2bool, help="Automatic Mixed Precision")
parser.add_argument("--num_workers", default=0, type=int, help="Workers for Datalodaer")
parser.add_argument("--seed", default=1, type=int, help="random seed")
parser.add_argument("--tqdm_bar", default=False, type=str2bool, help="Enable tqdm bar or not")

# ddp Settings
parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
parser.add_argument('--dist_backend', default='nccl', help='gloo or nccl ...')

# Model Settings
parser.add_argument("--n_layer", default=4, type=int, help="Number of Transformer Blocks")
parser.add_argument("--dp", default=0.0, type=float, help="Dropout Probability")
parser.add_argument("--if_ln", default=True, type=str2bool, help="If use LayerNorm or Not")
parser.add_argument("--vocab_size", default=-1, type=int, help="Vocab size, later updated while generating dataset")
parser.add_argument("--n_embd", default=512, type=int, help="Embedding Dimension")
parser.add_argument("--n_head", default=4, type=int, help="Number of Heads")
parser.add_argument("--block_size", default=512, type=int, help='maximum ctx length')
parser.add_argument("--act_name", default="relu", type=str, help="activation: relu, gelu, swiglu")
parser.add_argument("--widen_factor", default=4, type=int, help="MLP widening")
parser.add_argument("--mu", default=1.0, type=float, help="Skip connection strength")
parser.add_argument("--weight_tying", default=True, type=str2bool, help="If use weight tying")
parser.add_argument("--dont_decay_embd", default=False, type=str2bool, help="True means weight decay is not applied to Embedding layer")
parser.add_argument("--s", default=0.0, type=float, help="s=0 for SP, 1 for muP like attention. Use 0.0 only for now.")

# Data
parser.add_argument("--n_tasks_pl", default=64, type=int, help="number of independent tasks")
parser.add_argument("--n_tasks_rd", default=0, type=int, help="number of independent tasks")
parser.add_argument("--parallelogram", default=True, type=str2bool, help="Perform parallelogram construction on task vectors or not, n_tasks will be multiplied by 4 later")
parser.add_argument("--n_var", default=2, type=int, help="number of variables, x, y, z, ...")
parser.add_argument("--data_seed", default=0, type=int, help="random seed for generating datasets")
parser.add_argument("--split_data", default=True, type=str2bool, help="Train/Test set have different inputs or not.")
parser.add_argument("--data_pct", default=80.0, type=float, help="Data Percentage")
parser.add_argument("--split_tasks", default=False, type=str2bool, help="Alwasys Set to False, ood test is implemented seperately now")
parser.add_argument("--task_pct", default=50.0, type=float, help="Task Percentage, not used if split_tasks is set to False")
parser.add_argument("--p", default=29, type=int, help="Modulo p")
parser.add_argument("--base", default=29, type=int, help="Represent Numbers in base")
parser.add_argument("--n_point_per_row", default=32, type=int, help="Number of examples (x, y, ..., f(x, y, ...)) per row")
parser.add_argument("--encrypted", default=True, type=str2bool, help="Write the task vectors in seq or not. Keep True for ICL.")
parser.add_argument("--pos_hint", default=False, type=str2bool, help="Add positional hint for each digit or not")
parser.add_argument("--reverse_target", default=False, type=str2bool, help="Reverse the digits order of targets or not")
parser.add_argument("--show_mod", default=False, type=str2bool, help="Add <mod> <p> to token or not")
parser.add_argument("--show_seos", default=False, type=str2bool, help="Use <SOS> and <EOS> or not")
parser.add_argument("--fake_restart_steps", default=5000, type=int, help="Fake restart steps, to save memory")
parser.add_argument("--n_val_step", default=16, type=int, help="How many batches per evaluation")

# Optimization
parser.add_argument("--optim", default="adamw", type=str, help="optimizer, use adamw only")
parser.add_argument("--bs", default=1024, type=int, help="Batchsize for training")
parser.add_argument("--eval_bs", default=1024, type=int, help="Batchsize for evaluation")
parser.add_argument("--lr", default=1.5e-4, type=float, help="Learning Rate. We fix warmup initial lr to be 0.01 * lr, final lr to be 0.1 * lr")
parser.add_argument("--n_cycles", default=1, type=int, help="Cycles of scheduler, only use 1 cycle.")
parser.add_argument("--clip", default=0.0, type=float, help="Gradient clip, 0.0 means not used.")
parser.add_argument("--wd", default=2.0, type=float, help="Weight decay")
parser.add_argument("--beta1", default=0.9, type=float, help="Beta 1 for AdamW")
parser.add_argument("--beta2", default=0.98, type=float, help="Beta 2 for AdamW")
parser.add_argument("--eps", default=1e-8, type=float, help="Eps for AdamW")
parser.add_argument("--steps", default=200000, type=int, help="Training steps")
parser.add_argument("--warmup_steps", default=10000, type=int, help="Warmup steps")
parser.add_argument("--lr_decay", default='cosine', type=str, help="If Use Scheduler, cosine and linear are allowed")
parser.add_argument("--reshuffle_step", default=1, type=int, help="Keep to 1, old flag, no longer used.")
parser.add_argument("--fake_epochs", default=1000000, type=int, help="fake epochs, keep any large number")

# Saving
parser.add_argument("--steps_per_record", default=1000, type=int, help="Data Saving Frequency")
parser.add_argument("--n_ckpts", default=1, type=int, help="number of checkpoints to save along the trajectory")


"""Prepare Dataloader"""
class TrainDataset(Dataset):
    def __init__(self, dataset, bs, args):
        self.dataset = dataset.transpose(0, 1)
        self.n_data, self.n_task, self.dim = self.dataset.shape
        self.bs = bs
        self.n_point_per_row = args.n_point_per_row
        self.args = args

    def __len__(self):
        return self.bs * self.n_point_per_row * self.args.fake_restart_steps # To ensure we don't have to restart dataloader.

    def __getitem__(self, idx):
        step_x = self.dataset[idx % self.n_data] # (n_tasks, dim)
        return step_x


class EvalDataset(Dataset):
    def __init__(self, dataset, bs, args):
        self.dataset = dataset.transpose(0, 1)
        self.n_data, self.n_task, self.dim = self.dataset.shape
        self.bs = bs
        self.n_point_per_row = args.n_point_per_row
        self.args = args

    def __len__(self):
        return self.bs * self.n_point_per_row * (self.args.n_val_step + 5) # To ensure we don't have to restart dataloader.

    def __getitem__(self, idx):
        step_x = self.dataset[idx % self.n_data] # (n_tasks, dim)
        return step_x


def train_test_collate_fn(batch, args):
    inputs = torch.stack(batch, dim=1)  # (n_tasks, bs * ctx // n_tasks, dim)
    idx = torch.randperm(inputs.size(1))
    inputs = inputs[torch.arange(args.n_tasks)[:, None], idx[None, :]]
    targets = inputs.clone() # (n_tasks, bs * ctx_length, dim)
    targets[:, :, :-args.max_digits] = -100 # Mask the unpredictable part
    return inputs.view(-1, args.n_point_per_row * args.dim), targets.view(-1, args.n_point_per_row * args.dim)  # (bs, ctx_length * dim)


def ood_collate_fn(batch, args):
    inputs = torch.stack(batch, dim=1)  # (n_tasks, bs * ctx // n_tasks, dim)
    idx = torch.randperm(inputs.size(1))
    inputs = inputs[torch.arange(args.n_ood_tasks)[:, None], idx[None, :]]
    targets = inputs.clone() # (n_ood_tasks, bs * ctx_length, dim)
    targets[:, :, :-args.max_digits] = -100 # Mask the unpredictable part
    return inputs.view(-1, args.n_point_per_row * args.dim), targets.view(-1, args.n_point_per_row * args.dim)  # (bs, ctx_length * dim)


def get_train_dataloader(dataset, bs, args):
    custom_dataset = TrainDataset(dataset, bs=(bs * args.n_point_per_row // args.n_tasks) // args.ngpu_per_node, args=args)
    collate_fn = partial(train_test_collate_fn, args=args) # collate_fn in DataLoader should only take one input
    sampler = DistributedSampler(custom_dataset, shuffle=True, drop_last=False)
    dataloader = DataLoader(custom_dataset, batch_size=(bs * args.n_point_per_row // args.n_tasks) // args.ngpu_per_node, num_workers=args.num_workers, collate_fn=collate_fn, sampler=sampler, pin_memory=True)
    return dataloader, sampler


def get_eval_dataloader(dataset, bs, args):
    custom_dataset = EvalDataset(dataset, bs=(bs * args.n_point_per_row // args.n_tasks) // args.ngpu_per_node, args=args)
    collate_fn = partial(train_test_collate_fn, args=args) # collate_fn in DataLoader should only take one input
    sampler = DistributedSampler(custom_dataset, shuffle=True, drop_last=True)
    dataloader = DataLoader(custom_dataset, batch_size=(bs * args.n_point_per_row // args.n_tasks) // args.ngpu_per_node, num_workers=args.num_workers, collate_fn=collate_fn, sampler=sampler, pin_memory=True)
    return dataloader, sampler


def get_ood_dataloader(dataset, bs, args):
    custom_dataset = EvalDataset(dataset, bs=(bs * args.n_point_per_row // args.n_ood_tasks) // args.ngpu_per_node, args=args)
    collate_fn = partial(ood_collate_fn, args=args) # collate_fn in DataLoader should only take one input
    sampler = DistributedSampler(custom_dataset, shuffle=True, drop_last=False)
    dataloader = DataLoader(custom_dataset, batch_size=(bs * args.n_point_per_row // args.n_ood_tasks) // args.ngpu_per_node, num_workers=args.num_workers, collate_fn=collate_fn, sampler=sampler, pin_memory=True)
    return dataloader, sampler


# Main funtion
def main(args):

    # Init distributed mode
    init_distributed_mode(args)
    device = torch.device(args.device)

    if "cuda" in args.device:
        args.ngpu_per_node = torch.cuda.device_count()
        world_size = get_world_size()
        global_rank = get_rank()
        print(f'world_size {world_size}, global_rank {global_rank}')
    else:
        args.ngpu_per_node = 1

    print('device count: ', args.ngpu_per_node)

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
    args.max_ood_tasks = 256
    
    print(f'Parallelogram {args.parallelogram}, with total {len(pre_Ws)} task vectors: \n', pl_Ws, '\n rd: \n', rd_Ws)
    ood_Ws = get_ood_lists(pre_Ws, args)
    print(f'Ood with total {len(ood_Ws)} task vectors: \n', ood_Ws)

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
    train_set, val_set, tokenizer = prepare_data(args, pre_Ws)
    ood_train_set, ood_val_set, tokenizer = prepare_data(args, ood_Ws)

    args.vocab_size = tokenizer.__len__()
    print(f'vocab size: {args.vocab_size}')
    args.max_digits = tokenizer.max_digits
    args.max_digits = 2 * args.max_digits if args.pos_hint is True else args.max_digits
    args.dim = args.max_digits * (len(pre_Ws[0]) + 1)

    # Reshape to (n_tasks, n_examples_for_train/val, dim) shape

    train_set = train_set.view(args.n_tasks, len(train_set) // args.n_tasks, -1)
    val_set = val_set.view(args.n_tasks, len(val_set) // args.n_tasks, -1)
    ood_train_set = ood_train_set.view(args.n_ood_tasks, len(ood_train_set) // args.n_ood_tasks, -1)
    ood_val_set = ood_val_set.view(args.n_ood_tasks, len(ood_val_set) // args.n_ood_tasks, -1)

    print(f'train set shape: {train_set.shape}')
    train_iter, train_sampler = get_train_dataloader(train_set, args.bs, args)
    val_iter, _ = get_eval_dataloader(val_set, args.eval_bs, args)
    ood_train_iter, _ = get_ood_dataloader(ood_train_set, args.ood_bs, args)
    ood_val_iter, _ = get_ood_dataloader(ood_val_set, args.ood_bs, args)

    assert args.s == 0.0
    model = RoPETransformer(RoPEFlashAttention, args).to(device=device)

    ddp_model = DDP(model)
    compiled_ddp_model = torch.compile(ddp_model)

    scaler = GradScaler(init_scale=2**15, enabled=args.mixed_precision, growth_interval=1000)

    if args.n_ckpts == 0:
        ckpt_steps = None
    elif args.n_ckpts == 1:
        ckpt_steps = [args.steps]
    else:
        ckpt_steps = np.logspace(start=np.log10(args.warmup_steps), stop=np.log10(args.steps), num=args.n_ckpts, dtype=int)
        ckpt_steps = sorted(set(ckpt_steps))

    ckpt_prefix = f"../ckpts/noembd{args.dont_decay_embd}_parale{args.parallelogram}_pltask{args.n_tasks_pl}"
    data_prefix = f"../data/noembd{args.dont_decay_embd}_parale{args.parallelogram}_pltask{args.n_tasks_pl}"

    with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
        data = train_tf_icl(compiled_ddp_model, model,
                            train_iter, val_iter, ood_train_iter, ood_val_iter, train_sampler,
                            scaler, args, device,
                            ckpt_steps=ckpt_steps, ckpt_prefix=ckpt_prefix, data_prefix=data_prefix
                            )

    return

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
