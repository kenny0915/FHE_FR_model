# 排程入口

既有 Slurm 腳本保留原位置，避免改動 `sbatch` 指令與排程中的相對路徑。選擇研究家族時先查 [實驗索引](../experiments/README.md) 與 [設定路徑對照](../experiments/config_migration.json)。

- `slurm_train_*`：訓練與數值校準。
- `slurm_resume_*`：明確延續特定歷史輸出目錄，執行前核對 checkpoint。
- `slurm_eval_*`、`slurm_gate_*`：評估／數值 gate。
- `slurm_mine_*`、`slurm_screen_*`、`slurm_probe_*`：tail、替換位置與 activation 診斷。
- 新 degree-2 campaign 的 launcher 位於 [controlled_degree2/](../controlled_degree2/README.md)。

script 名稱不是完整 run 身份。提交前記錄實際 command、環境變數、config 快照、parent hash 與新的 run_id，stdout/stderr 放入對應 run 的 logs 目錄。此 checkout 不執行完整訓練。
