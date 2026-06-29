# python -m torch.distributed.run --nproc_per_node=4 train_v2.py

# iresnet50
# train
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
    --nproc_per_node=4 \
    train_v2.py configs/ms1mv3_r50

# test
CUDA_VISIBLE_DEVICES=0 python eval_ijbc.py \
  --model-prefix work_dirs/ms1mv3_r50/model.pt \
  --image-path ijb/IJBC \
  --result-dir work_dirs/ms1mv3_r50/ijbc_result \
  --batch-size 128 \
  --job ms1mv3_r50 \
  --target IJBC \
  --network r50

# poolformer
# train
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
    --nproc_per_node=4 \
    train_v2.py configs/ms1mv3_poolformer_s36

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
    --nproc_per_node=4 \
    train_v2.py configs/ms1mv3_poolformer_s36_speedup

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
    --nproc_per_node=4 \
    train_v2.py configs/ms1mv3_poolformer_s24_no_ln_no_gelu

# test
CUDA_VISIBLE_DEVICES=0 python eval_ijbc.py \
  --model-prefix work_dirs/ms1mv3_poolformer_s24_mlp2/model.pt \
  --image-path ijb/IJBC \
  --result-dir work_dirs/ms1mv3_poolformer_s24_mlp2/ijbc_result \
  --batch-size 128 \
  --job ms1mv3_poolformer_s24_mlp2 \
  --target IJBC \
  --network poolformer_s24_mlp2

CUDA_VISIBLE_DEVICES=0 python eval_ijbc.py \
  --model-prefix work_dirs/ms1mv3_poolformer_s36/model.pt \
  --image-path ijb/IJBC \
  --result-dir work_dirs/ms1mv3_poolformer_s36/ijbc_result \
  --batch-size 128 \
  --job ms1mv3_poolformer_s36 \
  --target IJBC \
  --network poolformer_s36
# mbf
# train
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
    --master_port=29500 \
    --nproc_per_node=4 \
    train_v2.py configs/ms1mv3_mbf

# test
CUDA_VISIBLE_DEVICES=0 python eval_ijbc.py \
  --model-prefix work_dirs/ms1mv3_mbf/model.pt \
  --image-path ijb/IJBC \
  --result-dir work_dirs/ms1mv3_mbf/ijbc_result \
  --batch-size 128 \
  --job ms1mv3_mbf \
  --target IJBC \
  --network mbf
