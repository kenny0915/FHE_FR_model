# Repository 整理紀錄：2026-09-24

本輪完成原規劃的第 2–5 階段。研究程式的數學與訓練策略維持原樣；唯一修復的既有 recipe 是原本含 conflict markers、無法匯入的 PoolFormer affine 設定。

## 2. 代表模型與來源鏈

- 登錄 7 個代表項目，生成 `experiments/runs/<run_id>/manifest.json`。
- 串流驗證 6 份可明確辨識的 checkpoint SHA-256：PReLU baseline、head-only v3、supervised01、progressive scaled/unscaled、run10 best；沒有載入權重或執行推論。
- failed controlled-direct-degree2 保留 14 個 non-finite augmented rows 的歷史證據；未能確認該次精確權重，所以 hash 保持 unknown。
- 從本機 config 與已保存的 calibration chain 擷取 97 筆明示的 parent/teacher/resume 引用。未知的父權重 hash 不臆測，外部路徑保留原值。
- run10 README 的 teacher=None 與 train config 的 teacher_weights 不一致，兩份原始內容均納入快照並標明衝突。

## 3. 結果匯入

- 60 列標準化結果：原先 36 列，加上 progressive scaled/unscaled 的 24 列。
- 324 列歷史 selection evidence 與新增 714 列本機 TAR CSV 分開保存，不重複合併計數。
- 本機 119 份 TAR CSV、45 份 config，以及 run10 README/config 共 166 份文字證據，包含 SHA-256，可離線重建。
- 額外欄位保存 failure policy、source-image failures、zeroed views、audit scope。scaled 21 failures / 42 zeroed views；unscaled 14 / 28，不能標成原始推論全數 finite。
- 189 個輸出群組補上證據覆蓋摘要。未審查的 protocol 仍明確標示待核對。

## 4. 文件入口

- 根 README 縮成研究／訓練／評估／交付入口，舊操作說明保留於 `docs/training.md`。
- 8 份根目錄研究報告移入 `reports/studies/`；原 ArcFace 文件移入 `docs/upstream/`。
- 更新文件相對連結，保留搬移對照表。歷史數據及原研究時點的結論不覆寫。

## 5. 程式與產物歸位

- 131 份設定實作依家族分到 `configs/` 子目錄，原檔保留相容入口。新舊 CLI 路徑與 Python imports 均可用。
- config loader 支援家族路徑；用 deepcopy 避免多次載入時污染 base config 或其他實驗。`output=None` 的輸出仍依原檔名推導。
- 逐份比較搬移前後的完整 resolved config，新舊兩種入口的 131 份設定均相同。衝突檔先依其檔頭設計解成 grouped fresh-start，再納入比對；原始衝突文字保留在 archive。
- 8 份壓縮檔、6 份排程 log 移入 `artifacts/`，共 117,464,743,434 bytes，透過同 filesystem rename 保持原檔案身份，留有可還原 journal。
- 未刪除任何權重或備份；尚無符合「確認依賴＋checksum 重複＋可驗證外部備份」的安全刪除證據。資料集與被引用的權重路徑保留。

## 驗證

- 14 個整理工具測試：結果來源、strict/nearest FAR、補零政策、離線重建、設定隔離、131 組相容入口，以及搬移碰撞／還原／中斷復原。
- 額外執行 10 個現有 recipe 測試，確認 reduced GELU、NL9/NL13、layerwise resume、Cheby8 與 Alpha7 的既有設定約束。
- 入口連結、CSV 重建、Python 編譯與 Git whitespace 檢查。
- 無完整訓練、無資料集推論、無加密推論實測。

## 留待研究判斷的事項

`needs_review` 表示歷史資料不足，不是程式整理尚未執行：舊 CSV 的 checkpoint 身份、部分評估 protocol、外部 run9 來源及未留存的訓練 commit，無法靠搬資料夾補回。新增研究請保存實際命令、resolved config、Git commit/dirty patch、精確 checkpoint hash 與完整評估政策。
