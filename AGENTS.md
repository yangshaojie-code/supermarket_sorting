# AGENTS.md

This repo is the **DG-202606 智慧零售** scoring client (ROS 2 Humble).

**Read first:** [`docs/AGENTS.md`](docs/AGENTS.md)（比赛、背景、导读，含本仓库 `docs/` 用法）  
**Then:** [`docs/DG-202606-AGENT-BRIEF.md`](docs/DG-202606-AGENT-BRIEF.md)（怎么改代码）

若 `vlm_pipeline/docs/` 里还有同名文件，以**本仓库 `docs/`** 为准。

Do not implement 文旅 (`vlm_pipeline` tasks) here. Do not rewrite `client_task_1.py` as the contest entrypoint. New behavior goes in `runtime/`.

`kind` names: `sanmingzhi heweidao shupian zhijin maidong kouxiangtang pingguo chengzi kele`

Tests: `python -m unittest discover -v` from this directory.
