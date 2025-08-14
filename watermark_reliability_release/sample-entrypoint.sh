wandb offline
export OPENAI_API_KEY=""
export WANDB_API_KEY=""
export CUDA_VISIBLE_DEVICES="0,1";
export HF_HOME="/workspace/cache"
export HF_ACCESS_TOKEN=""
huggingface-cli login --token $HF_ACCESS_TOKEN
export WANDB=T

### Experiment type ###
export RUN_GEN=T; export RUN_ATT=F; export RUN_EVAL=T; export DEBUG=F;
export LIMIT_ROWS=10000
#export METHOD="Redblock-rh"
export METHOD="Redblock-bh"
export BP_MAP="../balance_hash/map_freq.pkl"
### Generation ###
export MODEL_PATH="meta-llama/Llama-2-7b-hf";
#export MODEL_PATH="facebook/opt-1.3b"
#export MODEL_PATH="mistralai/Mistral-7B-v0.1"
#export MODEL_PATH="google/gemma-7b"
export BS=1; export TOKEN_LEN=200;
export D_NAME="allenai/c4"; export D_CONFIG="realnewslike";
export INPUT_FILTER="prompt_and_completion_length"
if [ $D_NAME == "lfqa" ]
then
  export INPUT_FILTER="completion_length"
fi
export NUM_BEAMS=1; export SAMPLING=T
export FP16=T; export MIN_GEN=10
## multi-bit
export MSG_LEN=8; export RADIX=4; export ZERO_BIT=F; export CODE_LEN=8;
export SEED_SCH="lefthash"; export GAMMA=0.25; export DELTA=1.175;
## logging
export RUN_NAME="${MSG_LEN}b-${TOKEN_LEN}T-${RADIX}R-${SEED_SCH}"
export OUTPUT_DIR="./experiments/bh-run"

### Attack ###
#export ATTACK_M="dipper";
#export ATTACK_M="gpt";
export DIPPER_ORDER=0; export DIPPER_LEX=20;
#export ATTACK_M="copy-paste";
export ATTACK_M="synonym";
export srcp="10%"; export CP_ATT_TYPE="single-single"
#export ATTACK_SUFFIX="dipper_lex=20"
export ATTACK_SUFFIX="synonym"
#export ATTACK_SUFFIX="gpt-3.5"

### Evaluation ###
export LOWER_TOL=25; export UPPER_TOL=25
export ORACLE_MODEL="meta-llama/Llama-2-13b-hf"
#export ORACLE_MODEL="facebook/opt-2.7b"
export IGNORE_R_NGRAM=T
export EVAL_METRICS="z-score"


mkdir -p ${OUTPUT_DIR}/log/${RUN_NAME}
bash ./run_pipeline.sh 2>&1 | tee -a ${OUTPUT_DIR}/log/${RUN_NAME}/output.log
cat ${OUTPUT_DIR}/log/${RUN_NAME}/output.log | grep "bit_acc"

