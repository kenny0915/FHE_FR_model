### iresnet50
# train
CUDA_VISIBLE_DEVICES=0,1 torchrun \
    --master_port=29500 \
    --nproc_per_node=2 \
    train_v2.py configs/casia_r50

# test
CUDA_VISIBLE_DEVICES=1 python eval_ijbc.py \
  --model-prefix work_dirs/casia_r50/model.pt \
  --image-path ijb/IJBC \
  --result-dir work_dirs/casia_r50/ijbc_result \
  --batch-size 256 \
  --job casia_r50 \
  --target IJBC \
  --network r50

CUDA_VISIBLE_DEVICES=1 python eval_ijbc.py \
  --model-prefix work_dirs/casia_r50_no_relu/model.pt \
  --image-path ijb/IJBC \
  --result-dir work_dirs/casia_r50_no_relu/ijbc_result \
  --batch-size 256 \
  --job casia_r50_no_relu \
  --target IJBC \
  --network r18_no_relu

CUDA_VISIBLE_DEVICES=1 python eval_ijbc.py \
  --model-prefix work_dirs/ms1mv3_r50_no_relu/model.pt \
  --image-path ijb/IJBC \
  --result-dir work_dirs/ms1mv3_r50_no_relu/ijbc_result \
  --batch-size 256 \
  --job ms1mv3_r50_no_relu \
  --target IJBC \
  --network r50_no_relu

## poolformer_s36
# train
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
    --nproc_per_node=4 \
    train_v2.py configs/ms1mv3_poolformer_s24_mlp2

# test
CUDA_VISIBLE_DEVICES=0 python eval_ijbc.py \
  --model-prefix work_dirs/ms1mv3_poolformer_s24_mlp2/model.pt \
  --image-path ijb/IJBC \
  --result-dir work_dirs/ms1mv3_poolformer_s24_mlp2/ijbc_result \
  --batch-size 128 \
  --job ms1mv3_poolformer_s24_mlp2 \
  --target IJBC \
  --network poolformer_no_ln_s24

### patch cnn
CUDA_VISIBLE_DEVICES=0,1 torchrun \
 --nproc_per_node=1 \
 train_v2.py configs/casia_patch_cnn
