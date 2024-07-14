gpu_id=1
def=ce
atk=ce
obs=omaml_clip
dataset=cifar100
k_obs=1
k_learn=4
k_def=5

SUPERCLASSES=(
    1 2 3 4 5 6 7 8 9
    10 11 12 13 14 15 16 17 18 19
)

for superclass_id in ${SUPERCLASSES[@]}; do
    bash scripts/train/base_lto.sh \
        ${gpu_id} \
        ${obs} \
        ${def} \
        ${atk} \
        ${dataset} \
        ${k_obs} \
        ${k_learn} \
        ${k_def} \
        ${superclass_id} \
        OBSTRUCTOR.TRAINER.MAX_EPOCH 100 \
        EVALUATOR.FREQ_EVAL 20 \
        $@
done
