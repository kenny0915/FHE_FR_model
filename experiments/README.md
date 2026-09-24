# 實驗管理入口

目前採「新增索引、保留原路徑」方式整理。此目錄管理研究問題、run 與評估來源；不改變既有訓練程式的 import 或輸出路徑。

## 找實驗與結果

- [輸出目錄清冊](index.csv)：本機 `work_dirs/` 每個第一層目錄一列；它是 artifact group，不保證恰好對應一次訓練。子階段與檔案位置在 [完整 inventory](inventory.json)。
- [代表模型與來源鏈](representatives.json)：baseline、head-only v3、channelwise supervised01、progressive scaled、run10 與重要失敗對照。
- [標準化評估表](../reports/tables/evaluations.csv)：由既有 0915 metrics 匯入，單位為 TAR 百分比。每列為 checkpoint × evaluation × FAR rule × FAR。
- [歷史證據表](../reports/tables/historical_evidence.csv)：由 selection evidence 內保存的 CSV 匯入。保留原始 audit/log 證據；checkpoint、校準條件或 actual FAR 未確認時填 unknown，不作合格排名。
- [0915 詳細研究報告](../docs/0915_result/README.md)、[v7 來源鏈](../docs/0915_result/v7_lineage.md)、[部署包紀錄](../docs/experiments/progressive_release_20260921.md)。

`family_hint` 只由目錄名稱推測，`needs_review` 不表示失敗。空檔案清單只表示掃描未發現該類副檔名，不代表實驗沒有結果。inventory 不掃 tensorboard、symlink 或資料集，不計算大檔 hash；它不能當成完整備份清單或刪除依據。清冊是此 checkout 的快照，其他機器上的產物可能不同。

## 家族與維護方向

| 家族 | 研究問題／代表入口 | 維護方向 |
|---|---|---|
| baseline | [原始 R50 設定](../configs/ms1mv3_r50.py)：PReLU 準確度基準 | 長期保留 teacher、評估結果與精確 hash |
| reduced_nonlinearity | [NL9/NL13](../iresnet50_reduced_nonlinearity_experiment.md)：減少非線性位置 | 保留對照與各層替換記錄 |
| polynomial_conversion | [低階多項式報告](../iresnet50_low_degree_polynomial_practicality_report.md)、[hard containment](../layerwise_poly_hard_containment_report.md)：degree、區間與穩定性 | 歷史對照；run10 原件先保護 |
| controlled_degree2 | [訓練入口](../controlled_degree2/README.md)：固定 degree-2，研究轉換、tail 與 head recovery | 主要研究入口；shared/adaptive/recipe 是子路線，不宣稱同一方法 |
| channelwise_calibration | [校準研究](../docs/experiments/channelwise_ijbc96_20260914.md)：逐 channel、數值修復與 template geometry | 與未使用 IJBC 校準的結果分組 |
| deployment | [匯出工具](../tools/progressive_release/README.md)：scaling、邊界與交付驗證 | 分開記錄浮點、range check、實際 FHE |
| other_backbones | [PoolFormer](../poolformer_s24_reduced_gelu_experiment.md)、MobileFaceNet、Patch-CNN、NF | 探索線；尚未刪除或實體封存 |
| unclassified | 無法從名稱可靠推測 | 人工審核，勿自動歸入主線 |

## 重建表格

在 repo 根目錄執行，只需 Python 標準函式庫，不需要 GPU 或模型權重：

```bash
python tools/experiment_registry/registry.py
python -m unittest discover -s tests -p 'test_experiment_registry.py'
```

更新本機清冊才使用：

```bash
python tools/experiment_registry/registry.py --scan
```

預設重建表格不改 inventory。匯入器不重新推論、不重新計算 ROC，不把歷史數字當新實驗；CSV 保存來源 JSON 的 SHA-256。新增來源格式應加入明確 adapter 與測試，不能靠檔名猜 TAR 意義。手動修改生成的 CSV 會被覆寫；應修正來源或 representative metadata。

## 新實驗規範

以 `experiments/run_manifest.template.json` 為起點，在 `experiments/runs/<run_id>/manifest.json` 保存一次 run 的設定；run_id 建議 `YYYYMMDD_family_variant_seedNN_attemptNN`。恢復同一中斷工作可沿用 run_id，改動訓練條件則新增 run 並記 parent。

- 保存完整 resolved config、實際命令、環境、seed、Git commit；工作樹有改動時另存 patch 並記 hash，commit 本身不足以重現。
- parent checkpoint、teacher、最終實際評估 checkpoint 都記位置與 SHA-256。不要以 `best.pt` 名稱推定身份。
- 分開記 train、fit、calibration、selection、test 資料與重疊限制。未知道的欄位用 null/unknown，不補成零或 false。
- 多項式必記逼近目標、每層／channel 區間與係數檔、輸入分布、超界率；部署 clipping 與訓練 guard 分開。
- 評估記 alignment、normalization、flip、precision、BN folding、evaluator commit、失敗處理、原始 score/label hash。strict 與 nearest FAR 分開；nearest threshold 不得套用 strict threshold。
- non-finite 數量必須連同分母與 audit scope；缺少逐 module audit 不等於全圖通過。
- FHE 深度需說明計算口徑與加密邊界；degree-2 不代表整個模型深度為 1。未執行 CKKS/BTS 時不填實測耗時或精度。

## 保留與後續整理

現在保留所有實驗。報告引用權重、teacher、初始化祖先、代表失敗案例、原始 scores、設定與 log 優先備份。續訓中的工作保留 optimizer/RNG/scheduler state。只有確認下游依賴、外部備份與 hash 後，才考慮清除重複權重或壓縮檔。

下一階段逐筆核對 needs_review，補齊 parent/checkpoint/evaluation 關係，再遷移程式。`utils/utils_config.py` 目前只支援平面 configs 匯入，不能直接把設定搬進家族子目錄。release builder 也依賴固定權重路徑。根目錄歷史報告先由此索引導覽，避免一次移動破壞既有連結。
