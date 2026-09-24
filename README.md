# FHE-compatible Face Recognition Experiments

研究 iResNet50 等人臉辨識模型的多項式 activation、數值穩定性與 FHE 推論限制。完整訓練在 GPU server 執行；本 checkout 用於修改程式與輕量驗證。

## 找結果

- [實驗管理與家族索引](experiments/README.md)：189 個輸出群組、代表模型、來源鏈與保留規則。
- [結果比較表](reports/README.md)：依校準資料、FAR 規則、失敗處理與 audit 範圍區分。
- [0915 degree-2 分析](docs/0915_result/README.md)：有／無 IJBC 校準候選。
- [MS1MV3 range calibration 與 BTS 失敗模擬](docs/experiments/progressive_best_bts_ms1k_20260917/README.md)、[未縮放對照](docs/experiments/progressive_unscaled_comparison_20260919/README.md)。
- [歷史研究報告](reports/studies/README.md)、[文件總覽](docs/README.md)。

## 執行入口

| 工作 | 入口 |
|---|---|
| 安裝與資料準備 | [環境](environment.yml)、[安裝](docs/install.md)、[上游資料說明](docs/upstream/arcface.md) |
| Baseline／舊方法訓練 | [完整操作說明](docs/training.md)、[configs 家族目錄](configs/README.md) |
| Controlled degree-2 訓練 | [controlled_degree2/README.md](controlled_degree2/README.md) |
| 評估 | [docs/eval.md](docs/eval.md)、`eval_ijbc.py` |
| 推論交付 | [完整包](tools/progressive_release/README.md)、[最小推論包](tools/progressive_inference/README.md) |
| 重建結果表 | `python tools/experiment_registry/registry.py` |
| 新增實驗 | [manifest 範本](experiments/run_manifest.template.json) |

新 config 使用家族路徑，例如 `configs/baseline/ms1mv3_r50`；舊的 `configs/ms1mv3_r50` 仍可用，輸出位置保持相同。不要用同一輸出目錄覆蓋不同實驗。

## 目錄

`backbones/`、`utils/`、`eval/` 是共用程式；`configs/` 是家族化設定；`controlled_degree2/` 是低階多項式研究程式；`scripts/` 保留既有 Slurm 入口。

`experiments/` 保存實驗 metadata 與來源快照，`reports/` 保存生成表格與研究報告，`docs/` 保存操作與專題文件。`work_dirs/`、`outputs/`、`artifacts/` 是本機大型產物，不放進 Git。大型檔案整理記錄與還原方式見 [儲存管理](docs/storage.md)。

比較數字前先核對 checkpoint hash、資料使用、strict／nearest FAR、non-finite 與補零政策。浮點多項式推論和 BTS range gate 不是實際加密推論實測。
