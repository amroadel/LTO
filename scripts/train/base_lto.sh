gpu_id=$1

obs=$2
def=$3
atk=$4

dataset=$5
k_obs=$6
k_learn=$7
k_def=$8
k_atk=5

superclass_id=$9

CUDA_VISIBLE_DEVICES=${gpu_id} python3 main.py \
    --config-dataset configs/data/${dataset}.yaml \
    --config-learner-atk configs/learner/${atk}/atk.yaml \
    --config-obstructor configs/obstructor/${obs}/${dataset}.yaml \
    --config-learner-def configs/learner/${def}/def.yaml \
    EXP_NAME ${dataset}/${obs}-${def}-k${k_def} \
    SAVER.CKPT_ATK_DIR ${dataset}/${atk}-k${k_atk} \
    DATA.SPLIT.MAX_K_DEF ${k_def} \
    LEARNER.DATALOADER.K_SHOT ${k_atk} \
    OBSTRUCTOR.LEARNER.DATALOADER.K_SHOT ${k_learn} \
    OBSTRUCTOR.DATALOADER.K_OBSTRUCT ${k_obs} \
    DATA.RESTRICTED.SUPERCLASS_ID ${superclass_id} \
    ${@:10}