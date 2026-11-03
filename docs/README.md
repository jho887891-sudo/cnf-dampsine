# CNF-DampSine 实验文档包

本目录包含根据 Deep Research 报告整理的两份执行文档：

-   `SPEC/CNF_DampSine_SPEC.md`：研究范围、数学定义、四种方法、超参数、实验矩阵、接口和验收标准。
-   `IMPLEMENTATION_PLAN/IMPLEMENTATION_PLAN.md`：Codex
    可逐项执行的任务地图，包含文件路径、接口、测试和验收命令。
-   `SPEC/references/CNF-DampSine_Deep_Research_Report.pdf`：本 SPEC
    的原始 18 页 Deep Research 报告。

## 使用方式

将本目录解压到 CNF-DampSine 项目根目录旁边或仓库根目录。让 Codex
先读取：

1.  `SPEC/CNF_DampSine_SPEC.md`
2.  `IMPLEMENTATION_PLAN/IMPLEMENTATION_PLAN.md`

开发组织采用用户选择的
A：四种方法统一规划；公共接口完成后，CNF-Gaussian、CNF-SIREN、Fourier、SIREN-PINN
四条方法轨道并行实现。

此文档包仅包含 SPEC 与 Implementation
Plan，不包含已经实现或验证通过的实验代码。
