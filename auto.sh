echo "1. Do Computation"
# Your code
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
    --nproc_per_node=4 \
    train_v2.py configs/ms1mv3_mbf

echo "2. Delete Interactive Container"
twccli rm ccs -f -s 5956743
#<CCS_ID>：using "twccli ls css" to find
