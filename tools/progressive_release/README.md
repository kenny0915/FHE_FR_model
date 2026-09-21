# Progressive degree-2 face model：組員實驗包

主要模型：`checkpoints/student_scaled.pt`。這是 progressive/student_best
經 **MS1MV3 1,000 張、seed 42 校準**後的 residual graph 等價縮放版本。
模型保留 BN，**不是 BN-folded checkpoint**。本包可獨立載入，無須原始 checkout。

## 快速開始

建議 Python 3.10，於解壓後的資料夾執行：

```bash
python -m pip install -r requirements-inference.txt
python verify_bundle.py
python example.py
python example.py --image /path/to/aligned_face.png --device cuda --output embedding.pt
```

驗證環境為 PyTorch 2.1.2+cu118、NumPy 1.23.5、Python 3.10.20。
依賴清單的 PyTorch 版本可依電腦選擇 CPU／CUDA wheel；完整原實驗環境在
`environment.yml`。只有執行原始完整校準／IJB-C 評估才需要其中的 MXNet、OpenCV、
scikit-image 等額外套件。範例無圖片時是零張量 smoke test，不是準確度測試。

```python
import torch
from loader import load_model

model, metadata = load_model(device="cuda")  # 自動 eval()
# images：已對齊人臉，RGB float32，[N,3,112,112]，pixel/127.5-1
with torch.inference_mode():
    embeddings = model(images.cuda())        # 原始 [N,512] embedding
# 若實驗使用 cosine similarity，可在明文端做 L2 normalization。
```

圖片需先依五點 landmark 對齊，不能用單純 resize 代替。模型不包含偵測、對齊或
L2 normalization。`example.py` 只做單張原圖推論，沒有 flip test／template 聚合。

## 包內檔案

- `checkpoints/student_scaled.pt`：目前主要模型，標準 controlled degree-2 checkpoint。
- `loader.py`、`example.py`、`verify_bundle.py`：獨立載入、推論與完整性驗證。
- `controlled_degree2/`、`backbones/`、`eval/`、`utils/` 等：原 repository 的程式快照。
  `MANIFEST.json` 記錄來源 commit、每個檔案 SHA-256、checkpoint 對應。
- `docs/experiments/progressive_best_bts_ms1k_20260917/`：目前校準流程、抽樣清單、
  六處範圍、完整 IJB-C 失敗標註及準確度。
- `docs/experiments/progressive_unscaled_comparison_20260919/`：縮放前後對照。
- `docs/experiments/progressive_best_bts_scaling_20260917.md`：縮放公式與最初等價性驗證。
  其早期 IJB-C 校準僅為歷史紀錄；目前模型使用 MS1MV3 校準。

本包只附縮放後模型；未附原始未縮放 checkpoint、人臉資料集、訓練 optimizer state 或 FHE runtime／金鑰。
原文件的 `/work/...` 和 `work_dirs/...` 是原實驗 provenance，請按本包 checkpoint
路徑替換；原 Slurm 腳本的帳號、工作目錄與 Python 路徑也需按組員環境調整。

## 模型尺度與使用語意

六處 BTS input 為 block residual addition 後：
`layer1.2`、`layer2.3`、`layer3.3`、`layer3.7`、`layer3.11`、`layer4.1`。
每個 stage 共用尺度：
`[0.18875518698808677, 0.2592094453179323, 0.06956599620400201, 0.40212774779961513]`。
Stage 3 三處共用尺度，以保持 identity shortcut、不新增 scaling 節點。

二次式為 `c0+c1*x+c2*x²`，近似目標為 per-channel PReLU；區間為 checkpoint
中每個 channel 的 `[-lam_fit,lam_fit]`。係數與區間已隨 stage 尺度同步轉換；
degree=2，每個 activation 一層 ciphertext multiplication。推論不做 clipping。
最終 embedding 已恢復原尺度，不需另外乘回 scale。

`load_model` 回傳的模型本身**不會將失敗 embedding 歸零**，也未加入實際 BTS。
如需要單張原圖的範圍失敗過濾，可用 `example.py --zero-failed`。
完整 IJB-C 的規則是「原圖或 flip 任一方向：六處任一值超過 [-1,1]，或出現
non-finite，整張原圖的兩個 embedding 都歸零」，判定與歸零在明文評估端執行。
區間內的最大值是校準結果，不是對所有未見資料的上界保證。

這份 checkpoint 的等價性是在 **eval / frozen BN statistics** 下驗證。
`model.train()` 會啟用原訓練程式的 BN、polynomial clipping／penalty 行為；
其結果不屬於這次等價性保證。若微調，請另行設計訓練流程並重新校準／驗證範圍。
多項式係數預設 frozen，若要更新需明確設定 `coeffs.requires_grad_(True)`。

## 已驗證結果

完整 IJB-C：469,375 張、15,658,489 組配對。縮放模型有 21 張失敗
（14 張 non-finite + 7 張僅超界），依上述規則歸零。
嚴格 FAR≤1e-4 的 TAR=95.13729%，與原始未縮放 progressive/best 相同。
詳細 ROC、原始數字與失敗清單在附帶文件；這是明文範圍判定評估，未模擬 BTS 噪聲。

## 用自己的資料重新校準／評估

從解壓根目錄執行。資料集需自行提供；若要重做原始校準，還需另行取得
原始未縮放 progressive/student_best.pt（不在本包內），不可把已縮放模型當成同一個
原始起點。output-dir 請使用新目錄：

```bash
python -m controlled_degree2.calibrate_bts_ms1mv3 \
  --checkpoint /path/to/original/progressive/student_best.pt \
  --output-dir outputs/new_scaled --dataset-root /path/to/ms1m-retinaface-t1 \
  --images 1000 --seed 42 --target-absmax 0.8 --batch-size 64 --device cuda

CUDA_VISIBLE_DEVICES=0 python eval_ijbc.py \
  --model-prefix checkpoints/student_scaled.pt --image-path /path/to/IJBC \
  --result-dir outputs/ijbc --batch-size 256 --image-workers 8 \
  --job scaled --target IJBC --network r50_controlled_d2 \
  --bts-failure-report outputs/ijbc/bts
```

未縮放對照需要自行提供原始 checkpoint、使用不同輸出路徑，並加上
`--ignore-bts-range`。所有 hash 可用 `python verify_bundle.py` 重驗；
這個驗證只需 CPU，不需要 MS1MV3 或 IJB-C。
