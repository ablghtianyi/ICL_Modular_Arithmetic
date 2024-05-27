import os
import sys
paths_to_add = ["..", "../.."]
for path in paths_to_add:
    sys_path = os.path.relpath(path)
    if sys_path not in sys.path:  # Check to avoid duplicates
        sys.path.append(sys_path)

from typing import Iterable, Optional
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.utils.data.distributed import DistributedSampler
from contextlib import nullcontext

import pandas as pd
from tqdm import tqdm

from _src.ddp_utils import save_ckpt_on_master, save_data_on_master
from _src.scheduler import build_optim_sched

def max_digits(n, base):
    """Return the maximum number of digits for a given p and base"""
    if n < 0:
        return max_digits(-n, base)
    if n == 0:
        return 1
    return int(math.log(n, base)) + 1


@torch.inference_mode()
def val_loss_acc_icl(model: nn.Module, val_loader: Iterable, args, device, n_val_step: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """
    This version returns per task measures.

    Warning: Assume balanced batch

    """
    ctx = nullcontext() if 'mps' in args.device else torch.autocast(device_type=args.device, dtype=args.dtype, enabled=args.mixed_precision)
    model.eval()


    num_valid = torch.zeros(args.n_tasks, device=device, dtype=args.dtype)
    val_loss = torch.zeros(args.n_tasks, device=device, dtype=args.dtype)
    correct = torch.zeros(args.n_tasks, device=device, dtype=args.dtype)
    correct_last = torch.zeros(args.n_tasks, device=device, dtype=args.dtype)
    num_valid_last = torch.zeros(args.n_tasks, device=device, dtype=args.dtype)

    for t, (x, y) in enumerate(val_loader):
        if t >= n_val_step:
            break

        x = x[:, :-1].contiguous().to(device, non_blocking=True)
        y = y[:, 1:].contiguous().to(device, non_blocking=True)
        valid_mask = (y != -100) & (y < args.base)
        with ctx:
            logits = model(x)
            losses = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100, reduction='none')

        if not math.isfinite(losses.mean().item()):
            return val_loss.data.cpu().float().numpy(), val_acc.data.cpu().float().numpy(), last_acc.data.cpu().float().numpy()

        losses = losses.view(args.n_tasks, args.eval_bs // (args.n_tasks * args.ngpu_per_node), -1)
        val_loss += (args.dim * losses.mean(dim=(-2, -1)) / args.max_digits) # Take account for the ignored elements

        valid_predictions = torch.masked_select(torch.argmax(logits, dim=-1), valid_mask)
        valid_labels = torch.masked_select(y, valid_mask)

        correct_mask = (valid_predictions == valid_labels).view(args.eval_bs, -1)  # (eval_bs, ctx * max_digits)
        temp_mask = torch.zeros((correct_mask.size(0), correct_mask.size(1) // args.max_digits), device=device, dtype=args.dtype) # (eval_bs, ctx)

        for i in range(args.max_digits):
            temp_mask += correct_mask[:, i::args.max_digits]  # Add one if first digit of the example is correct, second digit, ...

        correct_mask = temp_mask // args.max_digits  # Correct only when all digits are correct
        correct_mask = correct_mask.view(args.n_tasks, -1)  # (n_tasks, ctx)
        valid_mask = valid_mask.view(args.n_tasks, -1)  # (n_tasks, ctx)
        correct_last += correct_mask[:, -1]
        num_valid_last += valid_mask[:, -1]
        correct += correct_mask.sum(-1)  # (n_tasks, )
        num_valid += valid_mask.sum(-1)  # (n_tasks, )

    # Calculate the accuracy
    val_acc = correct / num_valid
    val_loss = val_loss / n_val_step
    last_acc = correct_last / num_valid_last

    model.train()
    return val_loss.data.cpu().float().numpy(), val_acc.data.cpu().float().numpy(), last_acc.data.cpu().float().numpy()


@torch.inference_mode()
def ood_loss_acc(model: nn.Module, ood_iter: Iterable, args, device, n_val_step: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """
    This version returns per task measures.

    Warning: Assume balanced batch

    """
    ctx = nullcontext() if 'mps' in args.device else torch.autocast(device_type=args.device, dtype=args.dtype, enabled=args.mixed_precision)
    model.eval()


    num_valid = torch.zeros(args.n_ood_tasks, device=device, dtype=args.dtype)
    ood_loss = torch.zeros(args.n_ood_tasks, device=device, dtype=args.dtype)
    correct = torch.zeros(args.n_ood_tasks, device=device, dtype=args.dtype)
    correct_last = torch.zeros(args.n_ood_tasks, device=device, dtype=args.dtype)
    num_valid_last = torch.zeros(args.n_ood_tasks, device=device, dtype=args.dtype)

    for t, (x, y) in enumerate(ood_iter):
        if t >= n_val_step:
            break

        x = x[:, :-1].contiguous().to(device, non_blocking=True)
        y = y[:, 1:].contiguous().to(device, non_blocking=True)
        valid_mask = (y != -100) & (y < args.base)
        with ctx:
            logits = model(x)
            losses = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100, reduction='none')

        if not math.isfinite(losses.mean().item()):
            return ood_loss.data.cpu().float().numpy(), ood_acc.data.cpu().float().numpy(), last_acc.data.cpu().float().numpy()

        losses = losses.view(args.n_ood_tasks, args.ood_bs // (args.n_ood_tasks * args.ngpu_per_node), -1)
        ood_loss += (args.dim * losses.mean(dim=(-2, -1)) / args.max_digits) # Take account for the ignored elements

        valid_predictions = torch.masked_select(torch.argmax(logits, dim=-1), valid_mask)
        valid_labels = torch.masked_select(y, valid_mask)

        correct_mask = (valid_predictions == valid_labels).view(args.ood_bs, -1)  # (ood_bs, ctx * max_digits)
        temp_mask = torch.zeros((correct_mask.size(0), correct_mask.size(1) // args.max_digits), device=device, dtype=args.dtype) # (ood_bs, ctx)

        for i in range(args.max_digits):
            temp_mask += correct_mask[:, i::args.max_digits]  # Add one if first digit of the example is correct, second digit, ...

        correct_mask = temp_mask // args.max_digits  # Correct only when all digits are correct
        correct_mask = correct_mask.view(args.n_ood_tasks, -1)  # (n_ood_tasks, ctx)
        valid_mask = valid_mask.view(args.n_ood_tasks, -1)  # (n_ood_tasks, ctx)
        correct_last += correct_mask[:, -1]
        num_valid_last += valid_mask[:, -1]
        correct += correct_mask.sum(-1)  # (n_ood_tasks, )
        num_valid += valid_mask.sum(-1)  # (n_ood_tasks, )

    # Calculate the accuracy
    ood_acc = correct / num_valid
    ood_loss = ood_loss / n_val_step
    last_acc = correct_last / num_valid_last

    model.train()
    return ood_loss.data.cpu().float().numpy(), ood_acc.data.cpu().float().numpy(), last_acc.data.cpu().float().numpy()


def train_tf_icl(compiled_model: nn.Module, model: nn.Module, train_iter: Iterable, val_iter: Iterable, ood_train_iter: Iterable, ood_val_iter: Iterable, train_sampler: DistributedSampler | None, scaler: GradScaler, args, device, ckpt_steps: Optional[list] = None, ckpt_prefix: Optional[str] = None, data_prefix: Optional[str] = None) -> list:
    """
    Training Loop for Incontext learning
    """
    ctx = nullcontext() if 'mps' in args.device else torch.autocast(device_type=args.device, dtype=args.dtype, enabled=args.mixed_precision)
    total_steps = args.steps
    warmup_steps = args.warmup_steps
    optimizer, scheduler = build_optim_sched(compiled_model, total_steps, warmup_steps, args)

    # gen_digits = 2 * args.max_digits if args.pos_hint is True else args.max_digits
    if ckpt_steps is not None:
        assert ckpt_prefix is not None

    step = 0

    data = []
    run_acc = 0.0
    test_loss = 0.0
    test_acc = 0.0
    last_test_acc = 0.0
    ood_train_last_acc = 0.0
    ood_val_last_acc = 0.0
    compiled_model.train()

    with tqdm(total=total_steps, desc="Training", unit="step", disable=(not args.tqdm_bar)) as pbar:

        for e in range(args.fake_epochs):
            # fake_epochs here to lower memory usuage, not needed really.
            if train_sampler is not None:
                train_sampler.set_epoch(e)

            for t, (x, y) in enumerate(train_iter):

                x = x[:, :-1].contiguous().to(device, non_blocking=True)
                y = y[:, 1:].contiguous().to(device, non_blocking=True)
                # Reset Gradients
                optimizer.zero_grad(set_to_none=True)

                with ctx:
                    logits = compiled_model(x)
                    losses = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100, reduction='none')
                loss = losses[losses > 0.0].mean()

                # Check NaN
                loss_value = loss.item()
                if not math.isfinite(loss_value):
                    print("Loss is {}, stopping training".format(loss_value))
                    sys.exit(1)

                # Gradient Step
                scaler.scale(loss).backward()
                if args.clip > 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                # Compute Acc
                with torch.inference_mode():
                    valid_mask = (y != -100) & (y < args.base)
                    valid_predictions = torch.masked_select(torch.argmax(logits, dim=-1), valid_mask)
                    valid_labels = torch.masked_select(y, valid_mask)
                    correct_mask = (valid_predictions == valid_labels).view(args.n_tasks, -1)
                    valid_mask = valid_mask.view(args.n_tasks, -1)
                    run_acc_last = (correct_mask[:, -1] / valid_mask[:, -1]).mean().item()
                    run_accs = correct_mask.sum(-1) / valid_mask.sum(-1)
                    run_acc = run_accs.mean().item()

                # Update Status Bar
                current_lr = scheduler.optimizer.param_groups[0]['lr']
                pbar.update(1)  # update the tqdm progress bar
                pbar.set_postfix({"lr": f'{current_lr:.2e}',
                                    "loss": f'{loss.item():.2e}',
                                    "acc": f'{(100 * run_acc):.1f}%',
                                    "test_loss": f'{test_loss:.2e}',
                                    "test_acc": f'{100 * test_acc:.1f}%',
                                    "gen_acc": f'{100 * last_test_acc:.1f}%',
                                    "ood_train_last_acc": f'{100 * ood_train_last_acc:.1f}%',
                                    "ood_val_last_acc": f'{100 * ood_val_last_acc:.1f}%',
                                    }
                                    )
                step += 1

                # Evaluation
                if (step == 1) or (step % args.steps_per_record == 0):
                    if 'free' in args.optim.lower():
                        optimizer.eval()
                        
                    with torch.inference_mode():
                        valid_mask = (y != -100) & (y < args.base)
                        valid_predictions = torch.masked_select(torch.argmax(logits, dim=-1), valid_mask)
                        valid_labels = torch.masked_select(y, valid_mask)
                        correct_mask = (valid_predictions == valid_labels).view(args.n_tasks, -1)
                        valid_mask = valid_mask.view(args.n_tasks, -1)
                        run_acc_last = (correct_mask[:, -1] / valid_mask[:, -1]).mean().item()
                        run_accs = correct_mask.sum(-1) / valid_mask.sum(-1)
                        run_acc = run_accs.mean().item()
                        
                        test_losses, test_accs, last_test_accs = val_loss_acc_icl(compiled_model, val_iter, args, device, n_val_step=args.n_val_step)
                        ood_train_losses, ood_train_accs, ood_train_last_accs = ood_loss_acc(compiled_model, ood_train_iter, args, device, n_val_step=args.n_val_step)
                        ood_val_losses, ood_val_accs, ood_val_last_accs = ood_loss_acc(compiled_model, ood_val_iter, args, device, n_val_step=args.n_val_step)

                        test_loss = test_losses.mean()
                        test_acc = test_accs.mean()
                        last_test_acc = last_test_accs.mean()

                        ood_train_loss = ood_train_losses.mean()
                        ood_train_acc = ood_train_accs.mean()
                        ood_train_last_acc = ood_train_last_accs.mean()

                        ood_val_loss = ood_val_losses.mean()
                        ood_val_acc = ood_val_accs.mean()
                        ood_val_last_acc = ood_val_last_accs.mean()

                        losses = losses.view(args.n_tasks, args.bs // (args.n_tasks * args.ngpu_per_node), -1)
                        losses = (args.dim * losses.mean(dim=(-2, -1)) / args.max_digits)

                        data.append([step, loss.item(), run_acc, run_acc_last, test_loss, test_acc, last_test_acc, ood_train_loss, ood_train_acc, ood_train_last_acc, ood_val_loss, ood_val_acc, ood_val_last_acc,])

                        df = pd.DataFrame(data, columns=['step', 'tr_loss', 'tr_acc', 'tr_acc_last', 'test_loss', 'test_acc', 'last_test_acc', 'ood_train_loss', 'ood_train_acc', 'ood_train_last_acc', 'ood_val_loss', 'ood_val_acc', 'ood_val_last_acc'])
                        path = data_prefix + f'_{args.model_name}_p{args.p}_base{args.base}_row{args.n_point_per_row}_ntask{args.n_tasks}_nvar{args.n_var}_dsplit{args.split_data}_dfrac{args.data_pct:.1f}_{args.act_name}_n{args.n_embd}_h{args.n_head}_d{args.n_layer}_lctx{args.block_size}_I{args.seed}_dI{args.data_seed}_{args.optim}_bs{args.bs}_T{args.steps:d}_Tw{args.warmup_steps:d}_lr{args.lr:0.2e}_wd{args.wd:.2e}.tab'
                        save_data_on_master(df, path, sep='\t')
                    
                    if 'free' in args.optim.lower():
                        optimizer.train()
                        
                # Save checkpoints and Data
                if ckpt_steps is not None:
                    if step in ckpt_steps:
                        path = ckpt_prefix + f'_{args.model_name}_p{args.p}_base{args.base}_row{args.n_point_per_row}_ntask{args.n_tasks}_nvar{args.n_var}_dsplit{args.split_data}_dfrac{args.data_pct:.1f}_{args.act_name}_n{args.n_embd}_h{args.n_head}_d{args.n_layer}_lctx{args.block_size}_I{args.seed}_dI{args.data_seed}_{args.optim}_bs{args.bs}_t{step}_T{args.steps:d}_Tw{args.warmup_steps:d}_lr{args.lr:0.2e}_wd{args.wd:.2e}.pth'
                        save_ckpt_on_master(model.state_dict(), path)

                if step >= total_steps:
                    break

            if step >= total_steps:
                break

    return data