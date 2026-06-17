## Auto run and delete container: bash auto.sh
TWCC_CLI_CMD=/home/u8798807/.local/bin/twccli
#<USERNAME>：account u8798807

echo "1. Do Computation"
# Your code
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
    --nproc_per_node=4 \
    train_v2.py configs/ms1mv3_r50_no_relu

echo "2. Delete Interactive Container"
$TWCC_CLI_CMD rm ccs -f -s 5971104
#<CCS_ID>：using "twccli ls ccs" to find

## Auto run and delete container: bash auto.sh
TWCC_CLI_CMD=/home/u8798807/.local/bin/twccli
#<USERNAME>：account u8798807

echo "1. Do Computation"
# Your code
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
    --nproc_per_node=4 \
    train_v2.py configs/ms1mv3_poolformer_s24_no_ln

echo "2. Delete Interactive Container"
$TWCC_CLI_CMD rm ccs -f -s 5975154
#<CCS_ID>：using "twccli ls ccs" to find
