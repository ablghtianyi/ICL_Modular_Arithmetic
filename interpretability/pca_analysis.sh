P=29
DIM=512
DEPTH=2
NHEAD=4
N_VAR=2
BS=1024
EVAL_BS=512
STEPS=200000
WARM_STEPS=10000
LR=1.5e-4
WD=2.0
FREEZE_SET_ID=0
MASK_SET_ID=0
DATA_PCT=$1
NTASKS_PL=$2
NTASKS_RD=$3 
PLOT_MODE=$4 # select the analysis one wants to do
START_POS=$5 # 0-94
END_POS=$6 # START_POS - 94
TASK_ID=$7 # 0 - (p-1)^2
OP_MODE=$8 # scanx / scany, not always effective
PLOT_HEAD_IDX=${9}

# Build the command string
CMD="python pca_analysis.py --device='mps' --mixed_precision=False --dtype='float32' --num_workers=0 \
--n_tasks_rd=$NTASKS_RD --n_tasks_pl=$NTASKS_PL --parallelogram=True --n_var=$N_VAR --p=$P --base=$P --data_pct=$DATA_PCT --split_data=True \
--model='rope_decoder' --act_name='relu' --block_size=512 --n_embd=$DIM --n_layer=$DEPTH --n_head=$NHEAD \
--optim='adamw' --lr=$LR --wd=$WD --dont_decay_embd=False --weight_tying=True --lr_decay='cosine' --clip=0.0 \
--steps=$STEPS --warmup_steps=$WARM_STEPS --n_point_per_row=32 \
--bs=$BS --eval_bs=$EVAL_BS --seed=1 --data_seed=0 --reshuffle_step=1 --n_measure=1 --plot_mode=$PLOT_MODE \
--start_pos=$START_POS --end_pos=$END_POS --task_id=$TASK_ID --operate_mode=$OP_MODE --plot_head_idx=$PLOT_HEAD_IDX \
--savefig=True"

eval $CMD