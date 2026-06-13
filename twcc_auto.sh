## Auto run and delete container: bash auto.sh
TWCC_CLI_CMD=/home/u8798807/.local/bin/twccli
#<USERNAME>：account u8798807

echo "1. Do Computation"
# Your code
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
    --master_port=29500 \
    --nproc_per_node=1 \
    train_v2.py configs/ms1mv3_mbf

echo "2. Delete Interactive Container"
$TWCC_CLI_CMD rm ccs -f -s 5956743
#<CCS_ID>：using "twccli ls css" to find
