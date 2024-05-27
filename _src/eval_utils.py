from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sympy import Symbol, Matrix, linsolve, mod_inverse


@torch.inference_mode()
def measure_perpos_accloss(model: nn.Module, val_loader: DataLoader, args, device, n_measure: int = 10, auxiliary = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Measure per position accuracy for one batch, modular arithmetic
    """
    ctx = nullcontext() if 'mps' in args.device else torch.autocast(device_type=args.device, dtype=args.dtype, enabled=args.mixed_precision)
    
    model.eval()
    
    acc_record = torch.zeros((args.eval_bs, args.n_point_per_row), device=device, dtype=args.dtype)
    loss_record = torch.zeros((args.eval_bs, args.n_point_per_row), device=device, dtype=args.dtype)
    
    for t, (x, y) in enumerate(val_loader):
        original_x = x.clone()
        x = x[:, :-1].contiguous().to(device)
        y = y[:, 1:].contiguous().to(device)
        if t >= n_measure:
            break

        with ctx:
            logits = model(x)
            losses = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction='none')
        losses = losses.reshape(logits.size(0), -1)
        pred = logits.argmax(-1)
        
        correct_mask = (pred[:, (args.dim-2)::args.dim] == y[:, (args.dim-2)::args.dim])
        
        acc_record += correct_mask
        loss_record += losses[:, (args.dim-2)::args.dim]

    if auxiliary is False:
        return acc_record.cpu().float().numpy() / n_measure, loss_record.cpu().float().numpy() / n_measure
    else:
        return acc_record.cpu().float().numpy() / n_measure, loss_record.cpu().float().numpy() / n_measure, original_x.cpu().float().numpy()


@torch.inference_mode()
def measure_perpos_accloss_single_batch(model: nn.Module, batch: tuple, args, device, n_measure: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """
    Measure per position accuracy for one batch, modular arithmetic
    """
    ctx = nullcontext() if 'mps' in args.device else torch.autocast(device_type=args.device, dtype=args.dtype, enabled=args.mixed_precision)
    
    model.eval()
    
    acc_record = torch.zeros((args.eval_bs, args.n_point_per_row), device=device, dtype=args.dtype)
    loss_record = torch.zeros((args.eval_bs, args.n_point_per_row), device=device, dtype=args.dtype)
    
    x, y = batch
    with ctx:
        logits = model(x)
        losses = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction='none')
    losses = losses.reshape(logits.size(0), -1)
    print(losses)
    pred = logits.argmax(-1)
    
    correct_mask = (pred[:, (args.dim-2)::args.dim] == y[:, (args.dim-2)::args.dim])
    
    acc_record += correct_mask
    loss_record += losses[:, (args.dim-2)::args.dim]

    return acc_record.cpu().float().numpy() / n_measure, loss_record.cpu().float().numpy() / n_measure


@torch.inference_mode()
def record_preds_and_targets(model: nn.Module, val_loader: DataLoader, args, device, top_k: int = 2, n_measure: int = 10) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Measure per position accuracy for one batch, modular arithmetic
    """
    ctx = nullcontext() if 'mps' in args.device else torch.autocast(device_type=args.device, dtype=args.dtype, enabled=args.mixed_precision)
    
    model.eval()
    
    preds = torch.zeros((args.n_measure, args.eval_bs, args.n_point_per_row, top_k), device=device, dtype=torch.long)
    targets = torch.zeros((args.n_measure, args.eval_bs, args.n_point_per_row), device=device, dtype=torch.long)
    seqs = torch.zeros((args.n_measure, args.eval_bs, args.dim * args.n_point_per_row), device=device, dtype=torch.long)
    for t, (x, y) in enumerate(val_loader):
        original_x = x.clone()
        x = x[:, :-1].contiguous().to(device)
        y = y[:, 1:].contiguous().to(device)
        if t >= n_measure:
            break

        with ctx:
            logits = model(x)
        pred = logits.topk(dim=-1, k=top_k)[1]
        preds[t, :, :, :] = pred[:, (args.dim-2)::args.dim, :]
        targets[t, :, :] = y[:, (args.dim-2)::args.dim]
        seqs[t, :, :] = original_x
    
    return preds.cpu().numpy(), targets.cpu().numpy(), seqs.cpu().numpy(), logits.cpu().numpy()


@torch.inference_mode()
def get_logits_and_loss(x, y, model, p, ctx):
    with ctx:
        logits = model(x)[:, :, :p]
        losses = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction='none')
    losses = losses.reshape(logits.size(0), -1)
    pred = logits.argmax(-1)
    logits = logits
    
    return losses, pred, logits


@torch.inference_mode()
def measure_grid_accloss_new(model: nn.Module, dataset:torch.Tensor, args, device, n_measure: int = 10, auxiliary = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Measure per position accuracy for one batch, modular arithmetic
    """
    if 'cuda' in args.device:
        device_type = 'cuda'
    elif 'cpu' in args.device:
        device_type = 'cpu'
    ctx = nullcontext() if 'mps' in args.device else torch.autocast(device_type=device_type, dtype=args.dtype, enabled=args.mixed_precision)
    
    model.eval()
    
    acc_record = np.zeros((args.eval_bs, dataset.shape[1]), dtype=float)
    loss_record = np.zeros((args.eval_bs, dataset.shape[1]), dtype=float)
    logit_record = np.zeros((args.eval_bs, dataset.shape[1], args.p), dtype=float)
    pred_record = np.zeros((args.eval_bs, dataset.shape[1]), dtype=float)

    input = dataset.clone()
    target = dataset.clone()
    for i in range(args.n_point_per_row):
        target[:, :, args.dim * i : args.dim * (i+1) - args.max_digits] = -100
    
    for t in range(dataset.shape[1]):
        x = input[:, t, :-1].contiguous().to(device=device)
        y = target[:, t, 1:].contiguous().to(device=device)
        
        losses, pred, logits = get_logits_and_loss(x, y, model, args.p, ctx)
        
        correct_mask = (pred[:, (args.dim-2)::args.dim] == y[:, (args.dim-2)::args.dim])
        # raise Exception(pred[:, (args.dim-2)::args.dim], y[:, (args.dim-2)::args.dim], correct_mask)
        
        acc_record[:, t] = correct_mask[:, -1].cpu().numpy()
        loss_record[:, t] = losses[:, (args.dim-2)::args.dim][:, -1].cpu().float().numpy()
        logit_record[:, t, :] = logits[:, (args.dim-2)::args.dim][:, -1].cpu().float().numpy()
        pred_record[:, t] = pred[:, (args.dim-2)::args.dim][:, -1].cpu().float().numpy()
        
    return acc_record, loss_record, logit_record, pred_record


################################################################################################
# Linear Solver
################################################################################################
def solve_equations_on_Zp(examples: np.ndarray, p):
    n_examples, dim = examples.shape
    coefficients = Matrix(examples[:, :dim - 1])
    constants = Matrix(examples[:, dim - 1])

    # Solve the matrix equation on Z_p
    solutions = []
    det = coefficients.det()
    if det % p != 0:
        det_inv = mod_inverse(det, p)
        result = det_inv * coefficients.adjugate() * constants
        result = result.applyfunc(lambda x: x % p)
        solutions.append(tuple(result))

    return solutions