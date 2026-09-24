# 本機產物管理

根目錄 8 份壓縮檔與 6 份 `adaptive-*` 排程 log 已分別移到 `artifacts/archives/` 與 `artifacts/scheduler_logs/`，共 117,464,743,434 bytes。這是同一 filesystem 的 rename，搬移前後檢查 device、inode、size、mtime；沒有讀取或重新壓縮 117 GB 內容，也沒有把 inode 檢查當成 SHA-256 驗證。

[搬移紀錄](../experiments/storage_migration.json) 保存原位置、新位置、檔案身份與保留決策。該紀錄只適用於此次本機 checkout，其他 clone 不應直接套用本機 inode。沒有刪除任何 checkpoint、資料集或備份，也沒有釋放磁碟容量。尚未確認外部備份與重複內容的壓縮檔全部保留。

```bash
# 查看新發現的根目錄壓縮檔／排程 log，不修改檔案
python tools/experiment_registry/storage.py

# 在原 checkout 還原此次搬移；有衝突或檔案身份改變時拒絕覆寫
python tools/experiment_registry/storage.py --restore

# 再套用 journal 的搬移
python tools/experiment_registry/storage.py --apply
```

工具先檢查全部檔案，寫 journal，再逐筆 rename；可恢復 rename 後、journal 更新前的中斷。Git 只保存工具與 metadata，不包含大型檔案。搬移後根目錄原壓縮檔路徑不再存在；程式／排程腳本未發現引用這些壓縮檔，手動解壓請使用 journal 中的新位置。

資料集、`work_dirs/`、`outputs/` 暫維持既有位置，因訓練、歷史來源鏈和 release builder 仍依賴這些路徑。不要只為版面整齊而改動正在被引用的權重路徑。

## 保留規則

- 必留：報告引用的精確 checkpoint、teacher／parent、代表失敗案例、原始 scores 與 label hash、設定、來源鏈、log。
- 續訓工作：保留 optimizer、scheduler、RNG、分散式 rank states。
- run10：保留原始來源與區間 buffers。已驗證 best checkpoint hash；外部 run9 parent 仍未知，不宣稱完整可重現。
- 可再生快取：只有確認沒有工作使用才清理。此輪未移除使用者 cache。
- 重複權重／壓縮檔：須比對 checksum、列出所有依賴、驗證外部備份可讀，再做刪除；目前沒有符合完整證據的刪除候選。

`experiments/local_evidence.json` 保存 119 份 TAR CSV、45 份本機 config，以及 run10 README/config 的文字快照；這不能代替原始權重或資料備份。
