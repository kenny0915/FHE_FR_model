# train
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
    --nproc_per_node=4 \
    train_v2.py configs/ms1mv3_poolformer_s24_fully_gated_frozen_std_fp32

# test
CUDA_VISIBLE_DEVICES=1 python eval_ijbc.py \
  --model-prefix work_dirs/ms1mv3_poolformer_nf12_progressive_fp32/model_epoch_16.pt \
  --image-path ijb/IJBC \
  --result-dir work_dirs/ms1mv3_poolformer_nf12_progressive_fp32/ijbc_result \
  --batch-size 256 \
  --job ms1mv3_poolformer_nf12_progressive_fp32 \
  --target IJBC \
  --network poolformer_nf12

CUDA_VISIBLE_DEVICES=1 python eval_ijbc.py \
  --model-prefix work_dirs/ms1mv3_r50_layerwise_poly_group4_d2/model_epoch_03.pt \
  --image-path ijb/IJBC \
  --result-dir work_dirs/ms1mv3_r50_layerwise_poly_group4_d2/ijbc_result \
  --batch-size 256 \
  --job ms1mv3_r50_layerwise_poly_group4_d2 \
  --target IJBC \
  --network r50_layerwise_poly
