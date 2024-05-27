# HyperPMs
P=29
DIM=512
DEPTH=4
NHEAD=4
BS=1024
EVAL_BS=1024
STEPS=200000
LR=1.5e-4
WD=2.0
NTASKS_PL=64
NTASKS_RD=0

# Define log file
LOGFILE="local_history.log"

# Build the command string
CMD="python icl_grokking_balanced_local.py --device='mps' --mixed_precision='False' --dtype='float32' --num_workers=0 \
--n_tasks_rd=$NTASKS_RD --n_tasks_pl=$NTASKS_PL --parallelogram='True' --n_var=2 --p=$P --base=$P --data_pct=80.0 --split_data=True \
--model='rope_decoder' --act_name='relu' --block_size=512 --n_embd=$DIM --n_layer=$DEPTH --n_head=$NHEAD \
--lr=$LR --wd=$WD --lr_decay='cosine' --clip=0.0 \
--steps=$STEPS --warmup_steps=10000 --steps_per_record=1000 --fake_restart_steps=5000 --n_point_per_row=32 \
--bs=$BS --eval_bs=$EVAL_BS --seed=1 --data_seed=0 --reshuffle_step=1 \
--tqdm_bar=True --n_ckpts=1"

# Log the start of the command
echo -e "\n $(date), Starting command:" >> $LOGFILE
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
echo -e "\n $(date): Finished, STATUS: $STATUS" >> $LOGFILE
