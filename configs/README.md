# Experiment configs

設定實作已按家族分到子目錄。根目錄同名 Python 檔只作相容入口；請編輯子目錄版本。

- `baseline/`：原始 R50/R100 與通用 baseline。
- `reduced_nonlinearity/`：NL9/NL13、linear/selective 變體。
- `polynomial_conversion/`：HerPN、Pillar、PreciseReLU、layerwise 與數值恢復。
- `other_backbones/`：PoolFormer、MobileFaceNet、Patch-CNN、NF。

完整路徑對應見 [migration map](../experiments/config_migration.json)。舊入口、新入口和 Python import 都可用；輸出目錄仍以檔名推導，不因家族路徑改變。

```bash
# 在 GPU server 執行；此處僅為指令範例
 torchrun --nproc_per_node=16 train_v2.py configs/baseline/ms1mv3_r50
```

`get_config` 回傳獨立 deepcopy，避免同一 Python process 先載入某實驗後，下一個設定繼承上一次額外欄位。環境變數覆寫仍由原有設定處理。

`ms1mv3_poolformer_s24_fully_gated_affine_fp32` 原有未解衝突使其無法 import。整理後採檔頭所描述的 grouped fresh-start 設定：`resume=False`、輸出 `...fully_gated_affine_grouped_fp32`。原始衝突內容保存在 [archive](../archive/config_conflicts/ms1mv3_poolformer_s24_fully_gated_affine_fp32.py.json)，不以這次修復推定任何歷史 run 的實際設定。
