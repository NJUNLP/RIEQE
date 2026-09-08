set -x

export CUDA_VISIBLE_DEVICES=0,1,2,3
# export CUDA_VISIBLE_DEVICES=4,5,6,7
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TORCH_CUDA_ARCH_LIST="8.6"
# export VLLM_USE_V1=0

NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
echo "Using $NUM_GPUS GPUs"

EPOCH=6
LR=2e-6
# DATA_NAME=span-en-de-subtask-unit-tgt-new-scale-merge-empty0.3-1k # 37
# DATA_NAME=span-en-de-subtask-unit-tgt-new-merge-empty0.3-1k-new # 36
# DATA_NAME=span-en-mr-subtask-unit-tgt-new-scale-merge-empty0.3-1k-new # 31
# DATA_NAME=span-4mixed-subtask-unit-tgt-new-scale-merge-empty0.3-4k-new # 141
# DATA_NAME=span-4mixed-subtask-unit-tgt-new-scale-merge-empty0.3-1k-new # 35
# DATA_NAME=span-zh-en-subtask-unit-tgt-new-scale-merge-empty0.3-1k-new # 43
# DATA_NAME=span-en-ru-subtask-unit-tgt-new-scale-merge-empty0.3-1k-new # 29
DATA_NAME=span-en-mr-subtask-unit-tgt-new-scale-merge-empty0.1-1k-new # 19

TRAIN_DATA=/anonymous/path/verl/data/$DATA_NAME/train.parquet
TEST_DATA=/anonymous/path/verl/data/$DATA_NAME/test.parquet
# MODEL_PATH=/anonymous/path/model/Qwen3-4B-t2507-subtask_nonCOT-unit-tgt-4mixed-new-scale-2ep
# MODEL_PATH=/anonymous/path/model/Qwen3-4B-t2507-subtask_nonCOT-unit-tgt-3mixed-new-scale-2ep
# MODEL_PATH=/anonymous/path/model/Qwen3-4B-t2507-subtask_nonCOT-unit-tgt-4mixed-new-47k-2ep
MODEL_PATH=/anonymous/path/model/Qwen3-4B-t2507-mr-cpted-subtask_nonCOT-unit-tgt-4mixed-new-scale-2ep
# MODEL_PATH=/anonymous/path/model/Qwen3-4B-t2507-subtask_nonCOT-unit-tgt-en-de-new-2ep
# MODEL_PATH=/anonymous/path/model/Qwen3-4B-t2507-subtask_COT-unit-tgt-en-de-new-correct-2ep
# MODEL_PATH=/anonymous/path/model/Qwen3-4B-t2507-subtask_COT-unit-tgt-en-de-new-all-2ep

# MODEL_PATH=/anonymous/path/model/Qwen3-4B-Instruct-2507
MODEL_NAME=$(basename $MODEL_PATH)

# REWARD_PATH=/anonymous/path/verl/verl/utils/reward_score/qe_reasoning_fbeta.py
REWARD_PATH=/anonymous/path/verl/verl/utils/reward_score/qe_reasoning_mcc.py
# REWARD_PATH=/anonymous/path/verl/verl/utils/reward_score/qe_reasoning_mcc_nonCOT.py

OUTPUT_PATH=/anonymous/path/model/${MODEL_NAME}-grpo-$DATA_NAME-${EPOCH}ep-MCC
ROLLOUT_DATA_DIR=/anonymous/path/verl/rollout_${MODEL_NAME}_grpo-${DATA_NAME}_${EPOCH}ep-MCC
VALIDATION_DATA_DIR=$ROLLOUT_DATA_DIR-val
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    custom_reward_function.path=$REWARD_PATH \
    custom_reward_function.name=compute_score_char_level_subtask \
    data.train_files=$TRAIN_DATA \
    data.val_files=$TEST_DATA \
    data.train_batch_size=128 \
    data.max_prompt_length=1500 \
    data.max_response_length=3000 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=-0.001 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.max_model_len=10000 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.65 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=19 \
    trainer.test_freq=1000 \
    trainer.total_epochs=${EPOCH} \
    trainer.rollout_data_dir=$ROLLOUT_DATA_DIR \
    trainer.validation_data_dir=$VALIDATION_DATA_DIR \
    trainer.default_local_dir=$OUTPUT_PATH

