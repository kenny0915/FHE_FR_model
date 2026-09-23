# IJB-C 3043.jpg：前處理與精簡推論包驗證

原圖為 `ijb/IJBC/loose_crop/3043.jpg`，大小 **2740×3131（寬×高）**，
metadata 第 3043 列，zero-based source index=3042。

## 實際前處理

1. OpenCV 讀取原圖，取得 BGR uint8。
2. 從 `ijbc_name_5pts_score.txt` 讀取既有五點 landmark（雙眼、鼻尖、兩個嘴角）；
   這次沒有重新做人臉偵測或 landmark 偵測。
3. 用 `skimage.transform.SimilarityTransform.estimate`，將原圖 landmark 對齊
   標準 112×112 人臉座標。這個 similarity transform 包含平移、旋轉、等比例縮放。
4. `cv2.warpAffine(..., (112,112), borderValue=0)` 重採樣及取出對齊人臉，
   使用預設 bilinear interpolation，超出原圖的區域填 0。不是整張圖直接 resize。
5. BGR 轉 RGB，HWC 轉 CHW，轉 float32，pixel/127.5-1，增加 batch 維度。
   原 IJB-C 評估器使用等價的 `/255 → -0.5 → /0.5`；浮點逐步運算可能略有差異。
6. Flip test 是對「已對齊的 112×112」做水平翻轉，不是先翻原圖再用原 landmark 對齊。

原圖五點（像素座標，依 metadata 小數顯示）：

```text
(1147.004, 1399.722)
(1908.165, 1163.138)
(1793.064, 1729.961)
(1441.275, 2220.645)
(2059.911, 2029.070)
```

標準五點約為 `(38.2946,51.6963)`、`(73.5318,51.5014)`、
`(56.0252,71.7366)`、`(41.5493,92.3655)`、`(70.7299,92.2041)`。
精確 float32 座標、2×3 transformation matrix 與原圖 SHA-256 存在 `report.json`。

## 保存的結果

- `embedding_original_reference.txt`：原圖 raw 512 維 embedding（3 張一批中的一張）。
- `embedding_flip_reference.txt`：對齊後翻轉圖的 raw 512 維 embedding。
- `embedding_original_cli.txt`：精簡包 `inference.py` 單張原圖輸出的 512 個數值。
- `report.json`：前處理參數、模型與 ZIP hash、六處範圍與誤差。
- `align_3043.py`：使用記錄的 landmark 重現前處理；檢查原圖 SHA-256 避免用錯圖片。
- 驗證 ZIP 另附 `3043_aligned.png` 與 `3043_aligned_flip.png`，可直接交給精簡 inference。

三種測試輸入（零張量、3043 原圖、3043 翻轉圖）用相同 batch 比對：
精簡程式與 repository 程式的 embedding 及六個邊界逐值相同，最大差 **0**。

單張 CLI 與 3 張 batch 參考輸出的最大絕對差為 **1.1920928955078125e-6**。
兩個比較的 batch 設定不同；不要把「同 batch 差 0」理解為所有 batch／裝置／
預處理浮點運算方式都必須 bitwise identical。以上數值均為 CPU FP32。
所有檔案都是 raw model embedding，未 L2 normalization、flip 合併或 faceness 加權。

## 重現

額外前處理依賴：NumPy、OpenCV、scikit-image。原始 3043.jpg 需自行提供。

```bash
python align_3043.py /path/to/3043.jpg --output 3043_aligned.png
```

解壓精簡模型包後：

```bash
python inference.py --image /path/to/3043_aligned.png --output embedding.txt
```

精簡包的 `inference.py --image` **要求已對齊的 112×112 圖片**，
不會替原始大圖自動找 landmark 或對齊。這個驗證資料包不包含模型權重，
也不更改原本只有 5 個檔案的模型 ZIP。
