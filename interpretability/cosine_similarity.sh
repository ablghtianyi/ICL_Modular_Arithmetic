P=29
DIM=512
DEPTH=$1
NHEAD=4
N_VAR=2
# Set BS based on DEPTH
if [ $DEPTH -eq 4 ]; then
    BS=1536
    EVAL_BS=768
    SEED=1
    DATA_PCT=80.0
    NTASKS_PL=96
    NTASKS_RD=0
elif [ $DEPTH -eq 2 ]; then
    BS=1024
    EVAL_BS=512
    SEED=2
    DATA_PCT=60.0
    NTASKS_PL=128
    NTASKS_RD=0
else
    BS=1024  # default fallback
    EVAL_BS=512
fi

STEPS=200000
WARM_STEPS=10000
LR=1.5e-4
WD=2.0
FREEZE_SET_ID=0
MASK_SET_ID=0

END_POS=$2 # Last token position to be included in the analysis
TASK_ID=$3 # 0 - (p-1)^2

# Build the command string
CMD="python cosine_similarity.py --device='mps' --mixed_precision=False --dtype='float32' --num_workers=0 \
--n_tasks_rd=$NTASKS_RD --n_tasks_pl=$NTASKS_PL --parallelogram=True --n_var=$N_VAR --p=$P --base=$P --data_pct=$DATA_PCT --split_data=True \
--model='rope_decoder' --act_name='relu' --block_size=512 --n_embd=$DIM --n_layer=$DEPTH --n_head=$NHEAD \
--optim='adamw' --lr=$LR --wd=$WD --dont_decay_embd=False --weight_tying=True --lr_decay='cosine' --clip=0.0 \
--steps=$STEPS --warmup_steps=$WARM_STEPS --n_point_per_row=32 \
--bs=$BS --eval_bs=$EVAL_BS --seed=$SEED --data_seed=0 --reshuffle_step=1 --n_measure=1 \
--end_pos=$END_POS --task_id=$TASK_ID --savefig=True"

eval $CMD