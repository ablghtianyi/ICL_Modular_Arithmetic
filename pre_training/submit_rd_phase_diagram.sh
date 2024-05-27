#!/bin/bash
p=29
d_embd=512
depth=4
n_head=4
lr=1.0e-4
wd=2.0
seed=1

# Set the initial port number
port=14122

# Set the range of data_pct and pl_task values
data_pct_start=20.0
data_pct_end=80.0
data_pct_step=10.0

rd_task_values=(8 16 32 64 128 256 512)

# Loop over data_pct values
for data_pct in $(seq $data_pct_start $data_pct_step $data_pct_end); do
    # Loop over rd_task values
    for rd_task in "${rd_task_values[@]}"; do
        # Submit the job using sbatch
        sbatch run_icl.sh $port $p $d_embd $depth $n_head $lr $wd $data_pct 0 $rd_task $seed
        
        # Increment the port number
        ((port++))
    done
done
