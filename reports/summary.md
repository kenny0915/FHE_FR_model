# 代表結果總覽

此表由 registry 產生，僅選 strict FAR=1e-4 作導覽；不是跨條件排名。完整 FAR、hash 與來源在 [CSV](tables/evaluations.csv)。

| Run | TAR (%) | Calibration | Evidence scope | Failed sources | Zeroed views |
|---|---:|---|---|---:|---:|
| channelwise_template_supervised01_20260915 | 96.06279 | ijbc_images_and_labels_calibration_set_performance | observed_finite_full_module_audit | unknown | unknown |
| controlled_degree2_accuracy_recovery_head_only_v3_20260908 | 93.42435 | no_ijbc_calibration_in_verified_chain_but_historical_ijbb_overlap | observed_finite_augmented_embeddings_only | unknown | unknown |
| ms1mv3_r50 | 96.55366 | baseline_reference | observed_finite_full_module_audit | unknown | unknown |
| progressive_best_bts_ms1k_20260917 | 95.13729 | ms1mv3_1000_images_for_scaling_ancestor_history_see_lineage | failure_filtered_plaintext | 21 | 42 |
| progressive_best_unscaled_ijbc_20260919 | 95.13729 | no_new_calibration_in_this_evaluation_ancestor_history_see_lineage | failure_filtered_plaintext | 14 | 28 |

unknown 表示來源未提供該項計數；不代表 0。source image 與 augmented view 是不同分母。
0915 head-only 只保留 embedding audit；channelwise 的 IJBC 校準集數字不是隔離測試。
progressive 兩列包含失敗補零，且縮放版本額外施加 range gate；均為浮點評估，沒有實測 FHE。
run10 與失敗案例若未建立精確 checkpoint→evaluation 對應，不移入此表。
