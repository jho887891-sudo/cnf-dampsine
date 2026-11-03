# CNF-DampSine 实验 SPEC

**版本：** 1.0\
**状态：** 待审阅的实施规范\
**直接依据：**《CNF-DampSine：阻尼振荡函数重建与物理硬约束实验设计报告》（18
页）\
**开发组织：** 用户选择
A------四种方法统一规划；共享接口完成后，四种方法并行实现。

------------------------------------------------------------------------

## 0. 文档用途与依据边界

本 SPEC 将 Deep Research 报告转成 Codex
可执行的技术要求。它不是新的研究方向，也不是把 CNF 接入 FunDiff 的
FAE/DiT。

### 0.1 来源优先级

1.  本项目 Deep Research 报告：实验参数、实验矩阵、MVP
    和验收设计的直接依据。
2.  CNF 论文：*Neural Fields with Hard Constraints of Arbitrary
    Differential Order*，arXiv:2306.08943v2。
3.  CNF 官方代码：<https://github.com/zfc946/CNF>，重点核查
    `network.py`、`nbf.py`、`diff_utils.py`、`requirements.txt`。
4.  FunDiff
    数据代码：<https://github.com/sifanexisted/fundiff/blob/main/damp_sine/data_utils.py>
    和 `damp_sine/configs/autoencoder.py`。
5.  SIREN
    原论文与官方代码：<https://arxiv.org/abs/2006.09661>、<https://github.com/vsitzmann/siren>。

### 0.2 事实类型必须区分

实现、日志和最终报告中区分：

-   **论文/官方代码事实**：CNF
    的基函数、约束矩阵、可微线性求解、高阶自动微分等机制。
-   **FunDiff 数据事实**：目标函数、参数分布和传感器配置。
-   **Deep Research
    实验建议**：K、M、σ、网络宽度、训练步数、噪声等级等本任务配置。
-   **工程验收阈值**：例如条件数告警和单元测试误差阈值。这些是本项目工程标准，不得写成论文结论。

若报告配置与官方源码当前接口不一致，先记录差异，再适配官方实现；不得静默地把另一种算法称为原始
CNF。

------------------------------------------------------------------------

## 1. 研究目标

在 FunDiff `damp_sine`
数据分布上，独立比较四种函数表示方法在函数重建和物理约束下的表现：

1.  **CNF-Gaussian**：可学习 Gaussian kernel basis + CNF
    硬约束系数求解。
2.  **CNF-SIREN**：CNF 官方支持的 SIREN 神经基函数 + CNF
    硬约束系数求解。
3.  **Fourier Collocation**：固定 Fourier 谱基 + 硬配置求解。
4.  **SIREN-PINN**：直接 SIREN 函数表示 + ODE/边界软约束 Loss。

> 第四种方法以 Deep Research 报告的命名为准：**SIREN-PINN**，不是无物理
> Loss 的纯 SIREN-MLP。它是软物理约束基线。

### 1.1 核心研究问题

-   Gaussian 与 SIREN 神经基函数对 CNF 硬约束求解有何影响？
-   可学习基函数与固定 Fourier 基函数在相同 ODE/边界条件下有何差异？
-   CNF 在配置点上的硬约束精度如何，离开配置点后全区间 ODE residual
    如何变化？
-   基函数容量、anchor、Gaussian σ、SIREN ω0
    对矩阵条件数和重建精度有何影响？
-   观测稀疏和噪声会如何影响重建与物理一致性？

### 1.2 非目标

-   不将 CNF 集成到 FunDiff 的 FAE、DiT 或 diffusion pipeline。
-   MVP 不做未知物理参数辨识；γ、ω、b 作为 oracle 参数用于构造样本特定
    ODE 和边界值。
-   不把有限 collocation 点上的等式约束表述成全区间严格满足 ODE。
-   不将 SIREN-PINN 的 penalty loss 误称为硬约束。
-   不在主结果中用 Ridge/Tikhonov 替换直接硬约束求解。

------------------------------------------------------------------------

## 2. 数学问题定义

### 2.1 目标函数

\[ f(x)=A
e\^{-`\gamma `{=tex}x}`\sin`{=tex}(`\omega `{=tex}x)+b,`\qquad `{=tex}x`\in[0,1]`{=tex}.
\]

定义去偏置函数：

\[ q(x)=f(x)-b=A e\^{-`\gamma `{=tex}x}`\sin`{=tex}(`\omega `{=tex}x).
\]

FunDiff 原始数据分布：

  参数                    分布
  ----------------------- --------------------------------
  \(A\)                   (U(0.5,1.0))
  (`\gamma`{=tex})        (U(2,4))
  (`\omega`{=tex})        (U(6`\pi`{=tex},8`\pi`{=tex}))
  (b)（源码中为 shift）   (U(-0.5,0.5))
  \(x\)                   (\[0,1\])

### 2.2 物理 ODE

\[
q'(x)=Ae\^{-`\gamma `{=tex}x}`\left[-\gamma\sin(\omega x)+\omega\cos(\omega x)\right]`{=tex}\]

\[ q''(x)=Ae\^{-`\gamma `{=tex}x}
`\left[(\gamma^2-\omega^2)\sin(\omega x)-2\gamma\omega\cos(\omega x)\right]`{=tex}.
\]

因此：

\[ `\boxed{q''+2\gamma q'+(\gamma^2+\omega^2)q=0}`{=tex} \]

定义样本特定线性算子：

\[ `\boxed{
\mathcal L_{\gamma,\omega}
=\frac{d^2}{dx^2}
+2\gamma\frac{d}{dx}
+(\gamma^2+\omega^2)I
}`{=tex} \]

所有 ODE 约束作用于 (q=f-b)，不是直接作用于带偏置的 (f)。

### 2.3 MVP 边界条件

主配置使用两端 Dirichlet：

\[
q(0)=0,`\qquad `{=tex}q(1)=Ae\^{-`\gamma`{=tex}}`\sin`{=tex}(`\omega`{=tex}).
\]

这两个边界值使用样本生成时保存的真实参数。该设置属于 **oracle-parameter
hard-constraint experiment**，不能宣称模型从数据中发现了 γ、ω 或 b。

齐次 ODE 单独使用会允许零解 (q`\equiv0`{=tex})，所以 ODE-only
只能作为说明零解问题的消融/负对照，不作为主要重建配置。

### 2.4 右边界退化时的初值回退

若 (q(1)) 数值上接近零，报告建议切换到：

\[ q(0)=0,`\qquad `{=tex}q'(0)=A`\omega`{=tex}. \]

注意 (q'(0)=A`\omega`{=tex}) 的符号为正。

该回退在同一个 (x=0) 同时施加 value 与 derivative。报告指出，普通
Gaussian kernel 在同一 anchor
施加多个算子时可能形成重复基函数/病态矩阵，因此：

-   普通 `KernelBasis` Gaussian 路线不得静默接受重复 anchor。
-   回退分支必须使用官方实现可支持的 hypernetwork kernel /
    `hypernet_siren` 路线，或将该样本标记为普通 Gaussian 路线不适用。
-   退化判定阈值不是论文常数。通过 `right_bc_degenerate_rel_tol`
    配置，默认工程阈值建议为
    (10^{-8}`\times`{=tex}`\max`{=tex}(\|A\|,10^{-12}))，并写入 run
    config。该阈值仅用于工程分支选择。

------------------------------------------------------------------------

## 3. 数据协议

### 3.1 原始采样与数据包装

FunDiff autoencoder config 的正式传感器设置为
128；原生成器函数默认参数可能为 100。为与正式配置一致，本实验固定：

-   每条曲线生成 128 个均匀 sensor points。
-   观测点数 MVP 为 32。
-   dense evaluation grid 为 2048。
-   同时保存 (A,`\gamma`{=tex},`\omega`{=tex},b)。FunDiff 原生成器只返回
    x/y，本项目包装器必须额外保存参数。

不得 import 整个 FunDiff/JAX 包；直接复刻经核对的数据公式与分布，避免把
JAX 环境依赖带入 CNF PyTorch 工程。

### 3.2 数据规模

  -----------------------------------------------------------------------
  阶段                        Train                Val               Test
  -------------- ------------------ ------------------ ------------------
  MVP                  1 条固定曲线   不需要独立 split     同一曲线 dense
                                                          grid 验证数值链

  正式主实验                    800                100                100

  可选扩展规模                 1600                200                200
  -----------------------------------------------------------------------

Train/Val/Test 的参数组合必须独立采样，固定并保存 split seed
和参数文件，避免同一参数曲线跨 split 泄漏。

### 3.3 随机种子

MVP：42。

正式统计：42、123、2026、3407、7777。

每次运行设置并记录 Python、NumPy、PyTorch、CUDA 随机种子；启用确定性
cuDNN 设置并记录环境。

### 3.4 观测点与噪声

  设置               (N\_{obs})   相对噪声
  -------------- -------------- ----------
  MVP                        32         0%
  稀疏观测消融     16 / 32 / 64         0%
  Noise-low                  32         1%
  Noise-high                 32         5%

噪声定义：

\[
`\epsilon`{=tex}`\sim`{=tex}`\mathcal `{=tex}N`\left`{=tex}(0,(`\eta`{=tex},`\operatorname{std}`{=tex}(f))\^2`\right`{=tex}),
`\qquad `{=tex}`\eta`{=tex}`\in`{=tex}{0.01,0.05}. \]

噪声实验中，ODE 和解析边界继续使用 clean physics/clean boundary；noisy
observations 只进入 regression objective。禁止把 noisy observations
作为精确 hard equality 后宣称去噪鲁棒性。

### 3.5 参数已知与未知

-   MVP/P0/P1/P2：使用保存的真实 (`\gamma`{=tex},`\omega`{=tex},b) 构造
    ODE 和边界条件。
-   P2 OOD：测试超出训练分布的 (`\omega`{=tex}) 与
    (`\gamma`{=tex})，但样本真实参数仍提供给物理算子；这是 OOD
    physics-conditioned evaluation，不是未知参数估计。
-   未知参数联合估计不属于本报告已定义实验。不得在本轮 Spec
    中自行加入参数辨识结论。

------------------------------------------------------------------------

## 4. 统一硬约束系统

### 4.1 基函数表示

CNF 两种方法采用：

\[
q\_`\theta`{=tex}(x)=`\sum`{=tex}\_{k=1}\^{K}`\beta`{=tex}\_k`\Psi`{=tex}\_k(x;`\theta`{=tex}).
\]

基函数矩阵：

\[
`\Phi`{=tex}\_`\theta`{=tex}(x)`\in`{=tex}`\mathbb `{=tex}R\^{N`\times `{=tex}K}.
\]

### 4.2 内部 collocation 点

MVP 使用 (M=31) 个 Chebyshev interior points：

\[
x_m=`\frac12`{=tex}`\left`{=tex}(1-`\cos`{=tex}`\frac{m\pi}{M+1}`{=tex}`\right`{=tex}),
`\quad `{=tex}m=1,`\ldots`{=tex},M. \]

点必须严格位于 ((0,1))，不得与 0 或 1 重合。正式消融允许
uniform/random/Chebyshev 三种分布。

### 4.3 约束矩阵

对每个基函数构造：

\[
(`\mathcal `{=tex}L\_{`\gamma`{=tex},`\omega`{=tex}}`\Psi`{=tex}\_k)(x_m)
= `\Psi`{=tex}\_k''(x_m)+2`\gamma`{=tex}`\Psi`{=tex}\_k'(x_m)
+(`\gamma`{=tex}^2+`\omega`{=tex}^2)`\Psi`{=tex}\_k(x_m). \]

矩阵：

\[ A_f=
```{=tex}
\begin{bmatrix}
(\mathcal L\Psi_1)(x_1)&\cdots&(\mathcal L\Psi_K)(x_1)\\
\vdots&&\vdots\\
(\mathcal L\Psi_1)(x_M)&\cdots&(\mathcal L\Psi_K)(x_M)\\
\Psi_1(0)&\cdots&\Psi_K(0)\\
\Psi_1(1)&\cdots&\Psi_K(1)
\end{bmatrix}
```
\]

\[ g=\[0,`\ldots`{=tex},0,q(0),q(1)\]\^`\top`{=tex}. \]

MVP：

\[ M=31,`\quad `{=tex}K=M+2=33, \]

\[ A_f`\in`{=tex}`\mathbb `{=tex}R\^{33`\times33`{=tex}},`\quad`{=tex}
g`\in`{=tex}`\mathbb `{=tex}R\^{33`\times1`{=tex}},`\quad`{=tex}
`\beta`{=tex}`\in`{=tex}`\mathbb `{=tex}R\^{33`\times1`{=tex}}. \]

\[ `\boxed{A_f\beta=g}`{=tex} \]

### 4.4 求解与梯度

-   主路径使用 `torch.linalg.solve(A, g)`，不得显式计算 (A\^{-1})。
-   (`\beta`{=tex}) 每次前向/训练迭代重新求解，不加入 Adam 参数列表。
-   CNF optimizer 只更新基函数参数 (`\theta`{=tex})（以及配置允许学习的
    kernel 参数）。
-   高阶自动微分必须保留计算图：`create_graph=True`。
-   对 (`\Phi`{=tex}`\in`{=tex}`\mathbb `{=tex}R\^{N`\times `{=tex}K})
    的每一列分别计算导数，或使用等价的 batched
    Jacobian/vmap；不得对整个矩阵简单求和求梯度而把 K
    个基函数导数混成一个值。
-   训练主循环：构造当前基函数 → 构造 A/g → solve β → 预测观测点 → 计算
    data regression + condition regularizer → backward → 更新 θ →
    下一步重新 solve β。

### 4.5 条件数与失败处理

记录 (`\kappa`{=tex}\_2(A))、rank、singular values、solver residual。

  ---------------------------------------------------------------------------
  条件数                      状态                    行为
  --------------------------- ----------------------- -----------------------
  (\<10\^6)                   正常                    继续

  (10^6`\sim`{=tex}10^{10})   警告                    保留结果并记录，检查
                                                      σ/anchor/初始化

  (\>10\^{10}) 或 solve fail  不稳定候选              标记 numerical
                                                      failure，QR/SVD
                                                      仅用于诊断

  (\>10\^{12})                严重病态                不得作为有效主结果
  ---------------------------------------------------------------------------

以上阈值是本项目工程规则，不是 CNF 论文规定。

主训练 condition regularizer 起点：

\[ `\mathcal `{=tex}L\_{CNF} =`\mathcal `{=tex}L\_{data}
+10\^{-6}`\log`{=tex}`\kappa`{=tex}\_2(A). \]

消融权重：(0,10^{-8},10^{-6},10\^{-4})。

Tikhonov：

\[
`\beta`{=tex}\_`\lambda`{=tex}=(A^`\top `{=tex}A+`\lambda `{=tex}I)^{-1}A\^`\top `{=tex}g,
`\quad `{=tex}`\lambda`{=tex}`\in`{=tex}{10^{-10},10^{-8},10^{-6},10^{-4}}
\]

只能作为独立 `CNF-Tikhonov` 消融，不能覆盖严格直接求解主结果。

### 4.6 方阵与非方阵约束

MVP 和主要容量实验保持 (K=M+2)，使用方阵直接求解。

报告还提出固定 (K=33) 时扫描 (M=15,31,63)。其中 M=15
会形成欠定矩阵，M=63 会形成过定矩阵，不能调用
`torch.linalg.solve`。这组必须作为**矩形约束扩展实验**单独实现并标记：

-   欠定：用最小范数解/伪逆类方法，报告 rank 与 hard residual。
-   过定：用 least-squares/SVD 诊断可行性；若 residual
    不达硬约束阈值，必须标记 infeasible，不能称为严格硬约束成功。
-   矩形实验结果不得混入方阵 CNF 主结果。
-   不得静默将矩形系统转成 Ridge。

------------------------------------------------------------------------

## 5. 四种方法与首选超参数

### 5.1 方法一：CNF-Gaussian

\[
`\Psi`{=tex}*j(x)=`\kappa`{=tex}(`\phi`{=tex}*`\theta`{=tex}(a_j),`\phi`{=tex}*`\theta`{=tex}(x)),
`\qquad`{=tex}
q*`\theta`{=tex}(x)=`\sum`{=tex}\_{j=1}\^{33}`\beta`{=tex}\_j`\Psi`{=tex}\_j(x).
\]

MVP：

  参数                                首选值
  ----------------------------------- ----------------------------------
  Basis family                        官方 Gaussian `KernelBasis` 路线
  K                                   33
  Anchor                              每个约束一个 anchor，共 33 个
  Anchor positions                    31 个 Chebyshev interior + 0 + 1
  Initial Gaussian (`\sigma`{=tex})   0.05
  Encoder                             `PositionalEncoder`
  `n_freqs`                           3
  Encoder parameters                  learnable
  Optimizer                           Adam
  LR                                  (1`\times10`{=tex}\^{-3})
  Max steps                           5000
  Early stopping                      patience 500
  Condition weight                    (1`\times10`{=tex}\^{-6})

σ 扫描：0.01、0.025、0.05、0.1、0.2。Encoder 消融：identity / positional
encoder；`n_freqs=1,3,5`。

官方接口对 σ 是否作为可学习参数应按实际构造器核查；Spec 要求暴露
`learn_sigma` 配置。MVP 固定 σ=0.05，P1 增加 fixed-vs-learned sigma
消融。此项是本项目控制变量的工程选择。

Gaussian basis 参数由训练 loss 学习；(`\beta`{=tex}) 由线性系统求解。

### 5.2 方法二：CNF-SIREN

不是单个 SIREN 直接输出 q。SIREN 用于生成一组 neural bases，CNF 仍通过
(A_f`\beta`{=tex}=g) 硬求系数。

MVP：

  参数                  首选值
  --------------------- ---------------------------
  Official basis name   `hypernet_siren`
  K                     33
  Main SIREN width      21
  Hidden layers         1
  `first_omega_0`       30
  `hidden_omega_0`      30
  Optimizer             Adam
  LR                    (3`\times10`{=tex}\^{-4})
  Steps                 5000，扫描上限 10000
  Early stopping        patience 500

P1 扫描：width 21/32/64；hidden layers 1/2/3；(`\omega`{=tex}\_0)
10/20/30/40。`sepbasis_siren` 作为官方可用 basis family 消融，不替换 MVP
的 `hypernet_siren`。

必须沿用官方 SIREN 专门初始化。实现前核对 `network.py`
中真实类名、构造器参数、basis id 到 hypernetwork 参数的映射；不得假设
`sepbasis_siren` 与 `hypernet_siren` 具有相同输出和适用范围。

### 5.3 方法三：Fourier Collocation

固定谱基：

\[ q(x)=c_0+`\sum`{=tex}\_{n=1}\^{16}
`\left[a_n\sin(n\pi x)+b_n\cos(n\pi x)\right]`{=tex}. \]

总列数：

\[ K=1+2`\times16`{=tex}=33. \]

MVP： - 33 列固定 Fourier basis。 - 主最高频率 (16`\pi`{=tex})，步长
(`\pi`{=tex})。 - 固定公共频率库，不允许将每个样本真实 (`\omega`{=tex})
直接加入 basis。 - 解析计算基函数一阶、二阶导数。 - 使用与 CNF 相同的
ODE + Dirichlet 行结构构造 (A`\beta`{=tex}=g)。 -
单次直接求解；没有神经参数、Adam 或训练步数。

候选频率池：({0,`\pi`{=tex},2`\pi`{=tex},`\ldots`{=tex},40`\pi`{=tex}})。主配置选取常数列 +
1π 到 16π 的 sin/cos 成对列。压力测试最高频率：24π、40π；列数仍固定为
33，频率选择规则必须保存到 config。频率步长消融：π/2、π、2π。

Fourier collocation 是固定谱基硬约束对照。其系数由 ODE/BC collocation
系统决定，不应谎称它和训练型 CNF 使用了相同的 data-regression
优化过程；观测点用于统一数据协议和评价，方法信息来源差异须在结果解释中说明。

### 5.4 方法四：SIREN-PINN

直接输出：

\[ q\_`\theta`{=tex}(x)=`\operatorname{SIREN}`{=tex}\_`\theta`{=tex}(x).
\]

软约束目标：

\[ `\mathcal `{=tex}L= `\lambda`{=tex}*d`\mathcal `{=tex}L*{data}
+`\lambda`{=tex}*r`\mathcal `{=tex}L*{ODE}
+`\lambda`{=tex}*b`\mathcal `{=tex}L*{BC}. \]

\[ `\mathcal `{=tex}L\_{data}
=`\frac1{N_{obs}}`{=tex}`\sum`{=tex}*i(q*`\theta`{=tex}(x_i)-q_i)\^2 \]

\[ `\mathcal `{=tex}L\_{ODE} =`\frac`{=tex}1M`\sum`{=tex}\_j `\left[
q_\theta''(x_j)+2\gamma q_\theta'(x_j)
+(\gamma^2+\omega^2)q_\theta(x_j)
\right]`{=tex}\^2 \]

\[ `\mathcal `{=tex}L\_{BC} =\|q\_`\theta`{=tex}(0)\|\^2+
\|q\_`\theta`{=tex}(1)-q(1)\|\^2. \]

MVP：

  参数                   首选值
  ---------------------- ---------------------------
  Width                  64
  Hidden layers          3
  (`\omega`{=tex}\_0)    30
  Initialization         SIREN 专用初始化
  Optimizer              Adam
  LR                     (1`\times10`{=tex}\^{-4})
  (`\lambda`{=tex}\_d)   1
  (`\lambda`{=tex}\_r)   1
  (`\lambda`{=tex}\_b)   100
  Training steps         10000 起步，最多 20000
  Early stopping         patience 1000

P1 扫描 width 32/64/128、layers 2/3/4、(`\omega`{=tex}\_0)
10/20/30/40、(`\lambda`{=tex}\_r`\in`{=tex}{0.1,1,10,100})。

SIREN-PINN 的边界和 ODE 是 penalty loss，不保证 machine
precision；必须与 CNF/Fourier 硬约束方法明确区分。

------------------------------------------------------------------------

## 6. 训练策略与参数更新边界

### 6.1 CNF 两种方法

第一阶段逐函数训练。每条函数有自己的
(`\gamma`{=tex},`\omega`{=tex},b)、边界右端值和约束矩阵。

每次迭代：

1.  当前 encoder/neural basis 参数生成 (`\Phi`{=tex}\_`\theta`{=tex})。
2.  对 basis columns 计算 (`\Psi`{=tex},`\Psi`{=tex}',`\Psi`{=tex}'')。
3.  构造 ODE + BC 矩阵 (A_f) 和右端 (g)。
4.  `beta = torch.linalg.solve(A_f, g)`。
5.  计算观测点预测
    (`\hat `{=tex}q=`\Phi`{=tex}*`\theta`{=tex}(x*{obs})`\beta`{=tex})。
6.  计算 data MSE + condition regularizer。
7.  backward；梯度经过 solve 回到 basis 参数。
8.  optimizer 只更新 basis 参数；下一轮重新计算 β。

### 6.2 Fourier

构造固定 basis、解析导数、约束矩阵并直接 solve 一次。无 optimizer。

### 6.3 SIREN-PINN

每条函数单独训练 SIREN 权重；不使用 CNF 的 (`\beta`{=tex}) 求解器。

### 6.4 Shared-basis transfer（P3）

在 800 条训练函数上学习共享 basis inductive bias；测试时冻结
(`\theta`{=tex})，针对每条 unseen test function 重新构造 (A_f)
并只求解新 (`\beta`{=tex})。SIREN-PINN
不获得额外跨样本共享优势，除非作为独立扩展。

------------------------------------------------------------------------

## 7. 实验矩阵

不做四方法 × 全部超参数的笛卡尔积。按 P0--P3 分层执行。

### P0：四方法 MVP

  方法             Basis/容量    M   Nobs   Noise
  -------------- ------------ ---- ------ -------
  CNF-Gaussian           K=33   31     32       0
  CNF-SIREN              K=33   31     32       0
  Fourier                K=33   31     32       0
  SIREN-PINN         width=64   31     32       0

MVP：单条曲线、seed=42、128 sensors、2048 dense grid、clean data。

### P1：结构与容量

  ------------------------------------------------------------------------------------------------------
  实验                    方法                    设置
  ----------------------- ----------------------- ------------------------------------------------------
  Capacity                Gaussian / CNF-SIREN /  K=17/33/65，主方阵对应 M=15/31/63
                          Fourier                 

  Observations            四方法                  Nobs=16/32/64，K=33、M=31

  Collocation density     四方法                  报告原提议
                                                  K=33、M=15/31/63；必须走矩形系统扩展分支并标记可行性

  Point layout            CNF / Fourier           uniform / random / Chebyshev

  Gaussian sigma          CNF-Gaussian            0.01/0.025/0.05/0.1/0.2

  SIREN omega             CNF-SIREN + SIREN-PINN  10/20/30/40
  ------------------------------------------------------------------------------------------------------

### P2：鲁棒性

-   Noise-low：1%。
-   Noise-high：5%。
-   OOD frequency：报告建议可扩展至 5π--6π 与 8π--9π。
-   OOD decay：报告提出 OOD
    γ，但未给出明确数值区间；区间必须由研究者在配置中明确后才可运行，不能由
    Codex 静默编造。

P2 先做 20 条 smoke test；无 numerical failure 后再扩展到 100 条 test
functions。

### P3：Shared-basis transfer

CNF-Gaussian/CNF-SIREN：训练 split 学共享 basis，测试时冻结
basis，只求解新样本 β。

### 正式统计

每个主配置：100 test functions × 5 seeds。保存均值、标准差、中位数和
numerical failure 数。

------------------------------------------------------------------------

## 8. 评价指标

所有方法在独立 2048-point dense grid 上评价；CNF/Fourier 另报告
collocation 点指标。

-   RMSE
-   Relative L2
-   Max absolute error
-   Dense ODE RMS residual
-   Dense ODE max residual
-   Collocation ODE RMS residual
-   Left/right boundary absolute error
-   (E\_{hard,`\infty`{=tex}}=\|A`\beta`{=tex}-g\|\_`\infty`{=tex})
-   (`\kappa`{=tex}*2(A))、(`\log`{=tex}*{10}`\kappa`{=tex}\_2(A))、rank、singular
    values
-   Solver time、training time、inference time
-   Parameter count
-   CNF-Gaussian/CNF-SIREN derivative relative error (E\_{d1},E\_{d2})

必须区分 hard constraint matrix residual、boundary point error 和
dense-grid ODE residual。有限 collocation 点上的硬约束不等于全区间 ODE
精确成立。

### 8.1 必须输出的图

1.  四方法重建曲线（GT、prediction、observations）。
2.  点对点绝对误差。
3.  dense ODE residual（log scale）。
4.  左右边界局部放大。
5.  condition vs K。
6.  condition vs M。
7.  RMSE vs K。
8.  dense residual vs K。
9.  Gaussian σ sensitivity。
10. SIREN ω0 sensitivity。
11. β coefficient heatmap。
12. constraint matrix heatmap。
13. singular-value spectrum。
14. noise degradation。
15. train/solver time vs Relative-L2 Pareto。

------------------------------------------------------------------------

## 9. 工程结构与接口

``` text
CNF_DampSine/
├── configs/
│   ├── mvp.yaml
│   ├── gaussian.yaml
│   ├── cnf_siren.yaml
│   ├── fourier.yaml
│   └── siren_pinn.yaml
├── data/
│   ├── generator.py
│   ├── dataset.py
│   └── splits/
├── models/
│   ├── bases/
│   │   ├── gaussian_kernel.py
│   │   ├── hypernet_siren.py
│   │   └── fourier_basis.py
│   ├── cnf.py
│   └── siren_pinn.py
├── physics/
│   ├── damped_oscillator.py
│   ├── constraints.py
│   └── collocation.py
├── solvers/
│   ├── differentiable_solve.py
│   └── diagnostics.py
├── training/
│   ├── train_cnf.py
│   ├── train_pinn.py
│   └── early_stop.py
├── evaluation/
│   ├── metrics.py
│   ├── evaluate.py
│   └── plots.py
├── scripts/
│   ├── run_mvp.py
│   ├── run_sweep.py
│   └── aggregate.py
├── tests/
└── results/
```

### 9.1 最小接口

``` python
def sample_damp_sine(n_functions: int, n_sensors: int, seed: int) -> dict:
    # return x, y_clean, y_noisy, params=[A, gamma, omega, b]
    ...

def make_collocation_points(M: int, kind: str, seed: int) -> torch.Tensor:
    # return [M,1], strictly inside (0,1)
    ...

class Basis(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input [N,1], return Phi [N,K]
        ...

def evaluate_basis_derivatives(basis, x):
    # return Phi0, Phi1, Phi2; each [N,K], graph preserved
    ...

def build_hard_constraint_system(
    basis, x_interior, gamma, omega, boundary_spec
):
    # return A [M+2,K], g [M+2,1]
    ...

def solve_coefficients(A, g, method="direct"):
    # return beta [K,1] and diagnostics
    ...

def evaluate_dense(model, sample, n_grid=2048):
    # return x_dense, gt, prediction, ODE residual and boundary diagnostics
    ...
```

接口最终以官方 CNF 类的实际构造器为准；若需 wrapper，保持上述项目内部
API 不变。

------------------------------------------------------------------------

## 10. 环境与复现性

第一轮忠实参考环境：

-   Python 3.9
-   PyTorch 1.13.0
-   PyTorch Lightning 1.8.1
-   SciPy 1.9.3

后续现代化环境可选，但必须单独记录版本并验证与参考实现的数值一致性。

维护独立 `cnf_ref` 与 `fundiff_ref` 环境。CNF 工程不 import FunDiff
JAX/Flax package；只复刻数据公式。

每次 run 保存：

``` text
results/<run_id>/
├── config.yaml
├── metrics.csv
├── summary.json
├── environment.txt
├── git_commit.txt
├── train.log
├── checkpoint_best.pt
├── checkpoint_last.pt
├── predictions.npz
└── figures/
```

CNF 每 500 steps 保存一次，只保留 best + last；SIREN-PINN 每 1000
steps；Fourier 保存频率集合、A、β 和配置，不创建神经网络 checkpoint。

`predictions.npz` 保存 dense x、GT、prediction、ODE
residual、collocation indices、β；必要时保存 A 和 g。

------------------------------------------------------------------------

## 11. 验收标准

### 数学正确性

-   解析目标函数在 dense grid 上的 ODE residual 接近浮点误差。
-   解析一阶/二阶导数与自动微分结果一致。
-   Fourier 解析导数与解析参考一致。

### 维度正确性

-   Basis 输出 `[N,K]`。
-   MVP 约束矩阵 `[33,33]`。
-   MVP 右端 `[33,1]`。
-   β `[33,1]`。
-   dense prediction `[2048,1]`。

### 硬约束正确性

float64 单元测试：

\[ `\frac{\|A\beta-g\|_\infty}{\max(1,\|g\|_\infty)}`{=tex}
`\le 10`{=tex}\^{-8}. \]

该阈值是工程验收标准，不是论文指标。最终报告仍需记录 float32/float64
行为和真实数值残差。

### 训练正确性

-   β 不在 optimizer 参数中。
-   CNF backward 能将梯度传回 basis 参数。
-   高阶导数计算图未被截断。
-   CNF 每次迭代重新 solve β。
-   SIREN-PINN 的 data/ODE/BC loss 分项可单独记录。
-   Fourier 不创建 optimizer。

### 运行正确性

四种方法均可独立运行，且统一入口能启动完整 P0：

``` bash
python -m pytest -q
python scripts/run_mvp.py --config configs/mvp.yaml --methods all
```

每种方法必须生成配置、日志、metrics、预测数组和图像；失败也必须保存错误原因和
numerical_failure 标记。

------------------------------------------------------------------------

## 12. 风险与必须显式处理的情况

1.  **重复 anchor**：普通 Gaussian kernel
    在同一点施加多个算子可能导致重复
    basis/秩亏；初值回退分支必须切换适配 basis。
2.  **矩形约束矩阵**：P1 固定 K 改 M 会产生非方阵；不得调用 square solve
    或将不满足约束的 least-squares 结果冒充硬约束。
3.  **导数列混叠**：对 `[N,K]` basis matrix 求导必须逐列或 batched
    Jacobian。
4.  **噪声泄漏到硬约束**：noisy observations 只进入 regression loss。
5.  **oracle 参数误述**：已知 γ/ω/b 的结果不是参数发现。
6.  **连续域过度声称**：硬约束只在离散配置点成立；dense residual
    单独报告。
7.  **条件数失控**：保留 numerical failure，不允许静默 ridge 修复。
8.  **Fourier 信息预算差异**：Fourier collocation 通过 ODE/BC
    求系数，不使用 neural regression training；结果解释需说明。
9.  **OOD γ
    区间未定义**：报告未提供具体数值；在配置明确前不运行该子实验。

------------------------------------------------------------------------

## 13. 参考来源

-   Deep Research 主报告：本目录
    `references/CNF-DampSine_Deep_Research_Report.pdf`。
-   CNF v2：<https://arxiv.org/abs/2306.08943>
-   CNF 官方仓库：<https://github.com/zfc946/CNF>
-   CNF
    官方网络实现：<https://github.com/zfc946/CNF/blob/main/network.py>
-   CNF NBF 实现：<https://github.com/zfc946/CNF/blob/main/nbf.py>
-   CNF
    微分工具：<https://github.com/zfc946/CNF/blob/main/diff_utils.py>
-   CNF
    requirements：<https://github.com/zfc946/CNF/blob/main/requirements.txt>
-   FunDiff damp_sine
    数据生成：<https://github.com/sifanexisted/fundiff/blob/main/damp_sine/data_utils.py>
-   FunDiff autoencoder
    config：<https://github.com/sifanexisted/fundiff/blob/main/damp_sine/configs/autoencoder.py>
-   SIREN 论文：<https://arxiv.org/abs/2006.09661>
-   SIREN 官方仓库：<https://github.com/vsitzmann/siren>
