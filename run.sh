### iresnet50
# train
CUDA_VISIBLE_DEVICES=0 torchrun \
    --master_port=29500 \
    --nproc_per_node=1 \
    train_v2.py configs/casia_r50

# test
CUDA_VISIBLE_DEVICES=1 python eval_ijbc.py \
  --model-prefix work_dirs/casia_r50/model.pt \
  --image-path ijb/IJBC \
  --result-dir work_dirs/casia_r50/ijbc_result \
  --batch-size 128 \
  --job casia_r50 \
  --target IJBC \
  --network r50

## poolformer_s36
# train
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
    --nproc_per_node=4 \
    train_v2.py configs/casia_poolformer_s36

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
    --nproc_per_node=4 \
    train_v2.py configs/ms1mv3_poolformer_s36

# test
CUDA_VISIBLE_DEVICES=0 python eval_ijbc.py \
  --model-prefix work_dirs/casia_poolformer_s36/model.pt \
  --image-path ijb/IJBC \
  --result-dir work_dirs/casia_poolformer_s36/ijbc_result \
  --batch-size 128 \
  --job casia_poolformer_s36 \
  --target IJBC \
  --network poolformer_s36

## mbf
# train
CUDA_VISIBLE_DEVICES=0 torchrun \
    --master_port=29500 \
    --nproc_per_node=1 \
    train_v2.py configs/casia_mbf

# test
CUDA_VISIBLE_DEVICES=0 python eval_ijbc.py \
  --model-prefix work_dirs/casia_mbf/model.pt \
  --image-path ijb/IJBC \
  --result-dir work_dirs/casia_mbf/ijbc_result \
  --batch-size 128 \
  --job casia_mbf \
  --target IJBC \
  --network mbf
