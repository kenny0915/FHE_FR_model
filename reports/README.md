# 結果總表

先看 [代表結果總覽](summary.md)。管理與重建方式見 [實驗入口](../experiments/README.md)。所有 TAR 單位為百分比。

| 產物 | 用途 |
|---|---|
| [evaluations.csv](tables/evaluations.csv) | 5 個有明確身份的模型評估、60 列 FAR 結果；含 strict/nearest、calibration、failure policy、source/view 失敗計數 |
| [historical_evidence.csv](tables/historical_evidence.csv) | 0915 整理時保存的 324 列歷史結果與有限性證據 |
| [local_evidence.csv](tables/local_evidence.csv) | 本機 119 份原始 TAR CSV 展開為 714 列；未核對的 protocol、hash 與 finite_scope 填 unknown |
| [lineage.csv](tables/lineage.csv) | 97 筆設定明示的 checkpoint/teacher/resume 引用與 calibration source hash |
| [run_summary.csv](tables/run_summary.csv) | 189 個 artifact group 的證據覆蓋程度、家族提示與待核對狀態 |
| [studies/](studies/README.md) | 根目錄搬入的歷史研究報告 |

前三份表有重疊來源，不可直接串接後當獨立實驗計數。local 表是稽核清冊，不是額外 714 次訓練。`artifact_group`、`run_id`、`evaluation_id` 的粒度不同。

校準資料、評估協定、checkpoint hash、FAR rule、失敗處理及 finite_scope 應先匹配再比較。新部署模型不繼承祖先模型的 TAR。run10 的歷史 degree-4 準確度尚未建立可驗證的 evaluator/score/checkpoint 對應，因此不硬填進標準化表。

```bash
python tools/experiment_registry/registry.py
```

預設只從 Git 內的來源與快照重建，不需要 GPU、資料集或權重。要更新本機原始 CSV/config 快照才加 `--capture-local`（新目錄先加 `--scan`）。不得手改生成表。
