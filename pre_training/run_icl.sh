#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64536
#SBATCH --time=6:00:00
#SBATCH --job-name=icl
#SBATCH --error=err/%A_%a.err
#SBATCH --output=out/%A_%a.out
#SBATCH --gpus=a100:1

source ~/.bashrc
cd $SLURM_SUBMIT_DIR

# Get the number of nodes and GPUs per node
NNODES=$SLURM_NNODES
NPROC_PER_NODE=$SLURM_GPUS_ON_NODE
MASTER_PORT=$1
export OMP_NUM_THREADS=$((SLURM_GPUS_ON_NODE * 2))

# HyperPMs
P=$2
DIM=$3
DEPTH=$4
NHEAD=$5
N_VAR=2
BS=1024
EVAL_BS=1024
STEPS=200000
WARM_STEPS=10000
LR=$6
WD=$7
DATA_PCT=$8
NTASKS_PL=$9
NTASKS_RD=${10}
SEED=${11}

# Define log file
LOGFILE="icl_history.log"

# Build the command string
CMD="torchrun --nnodes=$NNODES --nproc_per_node=$NPROC_PER_NODE --master_port=$MASTER_PORT \
icl_grokking.py --device='cuda' --mixed_precision=True --dtype='bfloat16' --num_workers=8 \
--n_tasks_rd=$NTASKS_RD --n_tasks_pl=$NTASKS_PL --parallelogram=True --n_var=$N_VAR --p=$P --base=$P --data_pct=$DATA_PCT --split_data=True \
--model='rope_decoder' --act_name='relu' --block_size=512 --n_embd=$DIM --n_layer=$DEPTH --n_head=$NHEAD \
--optim='adamw' --lr=$LR --wd=$WD --dont_decay_embd=False --weight_tying=True --lr_decay='cosine' --clip=0.0 \
--steps=$STEPS --warmup_steps=$WARM_STEPS --steps_per_record=1000 --fake_restart_steps=5000 --n_point_per_row=32 \
--bs=$BS --eval_bs=$EVAL_BS --seed=$SEED --data_seed=0 --reshuffle_step=1 \
--tqdm_bar=False --dist_backend='nccl' --n_ckpts=1 --n_val_step=16"

# Log the start of the command
echo -e "\n $(date), SLURM Job ID $SLURM_JOB_ID: Starting command:" >> $LOGFILE
echo "$CMD" >> $LOGFILE

# Execute the command
eval $CMD

# Check the exit status of the last executed command
if [ $? -eq 0 ]; then
  STATUS="SUCCESS"
else
  STATUS="FAILURE"
fi

# Log the end of the command
echo -e "\n $(date), SLURM Job ID $SLURM_JOB_ID: Finished, STATUS: $STATUS" >> $LOGFILE
