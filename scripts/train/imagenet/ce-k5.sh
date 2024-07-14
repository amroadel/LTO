gpu_id=6
def=ce
atk=ce
obs=omaml_clip
dataset=imagenet
k_obs=1
k_learn=4
k_def=5

SUPERCLASSES=(
    1 2 3 4 5 6 7 8 9
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
