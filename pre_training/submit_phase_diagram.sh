#!/bin/bash
p=29
d_embd=512
depth=4
n_head=4
lr=1.5e-4
wd=2.0
seed=1

# Set the initial port number
port=24353

# Set the range of data_pct and pl_task values
data_pct_start=20.0
data_pct_end=80.0
data_pct_step=10.0

pl_task_values=(2 4 8 16 32 64 128) # Actual value saved in the end is multiplied by 4

# Loop over data_pct values
for data_pct in $(seq $data_pct_start $data_pct_step $data_pct_end); do
    # Loop over pl_task values
    for pl_task in "${pl_task_values[@]}"; do
        # Submit the job using sbatch
        sbatch run_icl.sh $port $p $d_embd $depth $n_head $lr $wd $data_pct $pl_task 0 $seed
        
        # Increment the port number
        ((port++))
    done
done
