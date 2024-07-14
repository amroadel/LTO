gpu_id=1
def=ce
atk=ce
obs=omaml_attr
dataset=celeba
k_obs=1
k_learn=4
k_def=5

SUPERCLASSES=(
    1 2 3 4 5 6 
    7 8 9 10 11 
)

for superclass_id in ${SUPERCLASSES[@]}; do
    bash scripts/train/base_attr.sh \
        ${gpu_id} \
        ${obs} \
        ${def} \
        ${atk} \
        ${dataset} \
        ${k_obs} \
        ${k_learn} \
        ${k_def} \
        ${superclass_id} \
        EVALUATOR.TYPE attribute \
        EVALUATOR.FREQ_EVAL 10 \
        $@
done
