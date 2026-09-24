# 結果總表

管理與重建方式見 [實驗入口](../experiments/README.md)。

- `tables/evaluations.csv`：明確來源的 0915 checkpoint 評估，每個 FAR 分列 strict / nearest 規則。這是歷史結果標準化，並非全 repo 最新排行榜。
- `tables/historical_evidence.csv`：歷史比較候選與原始有限性證據；尚未核對的條件保留 unknown，不能直接合併排名。

校準資料使用、評估協定、checkpoint hash 與 finite_scope 均應先匹配再比較。新部署模型不自動繼承祖先模型的 TAR。既有圖表與研究文章暫留 `docs/` 原路徑。
