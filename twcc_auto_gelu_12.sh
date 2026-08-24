## Auto run and delete container: bash twcc_auto.sh
TWCC_CLI_CMD=/home/u8798807/.local/bin/twccli
#<USERNAME>：account u8798807

echo "1. Do Computation"
# Your code
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
    --nproc_per_node=4 \
    train_v2.py configs/ms1mv3_poolformer_s24_gelu12_fp32

echo "2. Delete Interactive Container"
$TWCC_CLI_CMD rm ccs -f -s 6093659
#<CCS_ID>：using "twccli ls ccs" to find
