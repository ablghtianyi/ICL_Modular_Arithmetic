import math
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from schedulefree import AdamWScheduleFree


def param_groups_weight_decay(
        model: nn.Module,
        weight_decay=1e-5,
        no_weight_decay_list=()
):
    no_weight_decay_list = set(no_weight_decay_list)
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if param.ndim <= 1 or name.endswith(".bias") or name in no_weight_decay_list:
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]
    

def param_groups_weight_decay_noembd(
        model: nn.Module,
        weight_decay=1e-5,
        no_weight_decay_list=()
):
    no_weight_decay_list = set(no_weight_decay_list)
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if param.ndim <= 1 or name.endswith(".bias") or 'wte' in name or 'switch' in name or name in no_weight_decay_list:
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


def linear_warmup_decay_scheduler(optimizer, total_steps, warmup_steps, base_lr=1e-2, max_lr=1., min_lr=1e-1):
    # Ensure warmup_steps is not more than total_steps
    if warmup_steps > total_steps:
        raise ValueError("warmup_steps must be less than or equal to total_steps")
    
    warmup_lambda = lambda step: min(base_lr + step / max(1, warmup_steps) * (max_lr - base_lr), max_lr)
    decay_lambda = lambda step: max(max_lr - (step - warmup_steps) / max(1, total_steps - warmup_steps) * (max_lr - min_lr), min_lr)
    combined_lambda = lambda step: warmup_lambda(step) if step <= warmup_steps else decay_lambda(step)
    scheduler = LambdaLR(optimizer, lr_lambda=combined_lambda)
    return scheduler


def cyclic_linear_warmup_decay_scheduler(optimizer, total_steps, warmup_steps, n_cycles=1, base_lr=1e-2, max_lr=1.0, min_lr=1e-1):
    """Linear warmup and decay scheduler with cycles"""
    cycle_length = total_steps // n_cycles
    if warmup_steps > cycle_length:
        raise ValueError("warmup_steps must be less than or equal to cycle_length")

    def lr_lambda(current_step):
        cycle = current_step // cycle_length
        step_in_cycle = current_step % cycle_length
        
        if step_in_cycle <= warmup_steps:
            # Warmup phase for the current cycle
            return base_lr + step_in_cycle / max(1, warmup_steps) * (max_lr - base_lr)
        else:
            # Decay phase for the current cycle
            decay_steps = cycle_length - warmup_steps
            return max(max_lr - (step_in_cycle - warmup_steps) / max(1, decay_steps) * (max_lr - min_lr), min_lr)

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    return scheduler


def cyclic_cosine_warmup_decay_scheduler(optimizer, total_steps, warmup_steps, n_cycles=1, base_lr=1e-2, max_lr=1.0, min_lr=1e-1):
    """Linear Warmup with Cosine decay, with cycles"""
    cycle_length = total_steps // n_cycles
    
    def cosine_decay(step_in_cycle):
        cosine_decay_value = (1 + math.cos(math.pi * step_in_cycle / (cycle_length - warmup_steps))) / 2
        return min_lr + (max_lr - min_lr) * cosine_decay_value
    
    def lr_lambda(current_step):
        cycle = current_step // cycle_length
        step_in_cycle = current_step % cycle_length
        
        if step_in_cycle <= warmup_steps:
            # Warmup phase for the current cycle
            return base_lr + step_in_cycle / max(1, warmup_steps) * (max_lr - base_lr)
        else:
            # Cosine decay phase for the current cycle
            return cosine_decay(step_in_cycle - warmup_steps)
    
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    return scheduler


def build_optim_sched(model, total_steps, warmup_steps, args, base_lr=1e-2, max_lr=1., min_lr=1e-1):
    if args.dont_decay_embd == False or args.dont_decay_embd is None:
        no_decay, decay = param_groups_weight_decay(model, weight_decay=args.wd)
    elif args.dont_decay_embd == True:
        no_decay, decay = param_groups_weight_decay_noembd(model, weight_decay=args.wd)
        
    if args.optim.lower() == 'adamw':
        optimizer = optim.AdamW([no_decay, decay], lr=args.lr, betas=(args.beta1, args.beta2))
    elif args.optim.lower() == 'sgd':
        optimizer = optim.SGD([no_decay, decay], lr=args.lr, momentum=0.9)
    elif args.optim.lower() == 'adamwfree':
        optimizer = AdamWScheduleFree([no_decay, decay], lr=args.lr, betas=(args.beta1, args.beta2), warmup_steps=warmup_steps)
    else:
        raise Exception("Optimizer Not Implemented")
    
    # warmup_steps = args.warmup_steps
    
    if args.lr_decay.lower() == 'linear' or args.lr_decay == True:
        scheduler = cyclic_linear_warmup_decay_scheduler(optimizer, total_steps, warmup_steps, n_cycles=args.n_cycles, base_lr=base_lr, max_lr=max_lr, min_lr=min_lr)
    elif args.lr_decay.lower() == 'cosine':
        scheduler = cyclic_cosine_warmup_decay_scheduler(optimizer, total_steps, warmup_steps, n_cycles=args.n_cycles, base_lr=base_lr, max_lr=max_lr, min_lr=min_lr)
    elif args.lr_decay.lower() == 'none' or 'free' in args.optim.lower() or args.lr_decay == False:
        scheduler = LambdaLR(optimizer, lr_lambda=lambda step: 1.)
    return optimizer, scheduler