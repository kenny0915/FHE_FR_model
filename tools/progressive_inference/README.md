# 縮放後 progressive 模型：最小推論包

只有 5 個檔案：`model.pt`、`backbone.py`、`inference.py`、`requirements.txt`、本說明。
`model.pt` 即 MS1MV3 1,000 張校準的 student_scaled.pt，內容沒有修改。

建議 Python 3.10；PyTorch wheel 請依 CPU／CUDA 環境選擇。

```bash
pip install -r requirements.txt
python inference.py --image aligned_face.png --output embedding.txt
# GPU：加上 --device cuda
# 不給 --image 會使用零張量做 smoke test，不能當作準確度測試。
```

圖片必須已做人臉五點對齊且為 112×112；程式讀成 RGB，使用 pixel/127.5-1。
輸出是原圖的 512 維 float32 embedding，每行一個值；不做 flip 或 L2 normalization。

已有 tensor 推論流程可直接用：

```python
import torch
from inference import load_model
model = load_model(device="cuda")
# images: float32, [N,3,112,112], RGB, pixel/127.5-1
with torch.inference_mode():
    embeddings = model(images.cuda())  # [N,512]
```

模型保留 BN，PReLU 已全部換成 c0+c1*x+c2*x²。近似目標為 per-channel PReLU，
近似區間存在各層 lam_fit 中：[-lam_fit,lam_fit]。係數、BN 與 Conv 已縮放，
不需要再次 scaling；最終 embedding 已恢復原尺度。每個二次式一層乘法深度。

這是 **eval 推論程式**，沒有 clipping、訓練 loss、實際 BTS、範圍失敗歸零或資料集程式。
BN 請維持 eval；模型可能在少量特殊輸入產生 non-finite，CLI 會報錯而非輸出結果。
若要重現先前 IJB-C 的失敗處理，需自行檢查 layer1.2、layer2.3、layer3.3、layer3.7、
layer3.11、layer4.1 的 block 輸出；原圖／flip 任一超界或 non-finite，就將兩個 embedding 歸零。
一般 tensor API 回傳原始模型輸出，沒有自動過濾。
