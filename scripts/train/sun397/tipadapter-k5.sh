gpu_id=0
def=tipadapter
atk=tipadapter
obs=omaml_clip
dataset=sun397
k_obs=1
k_learn=4
k_def=5
SUPERCLASSES=(
    1 2 3 4 5 6 7 8 9
    10 11 12 13 14
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
        $@
done
