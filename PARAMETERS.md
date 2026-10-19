# 主实验全部参数与实现约定

来源：P = 本次 PDF/LaTeX 正文明确给出；S = 工作区已有补充方法，与正文一致但未包含在此次附件中；I = 正文未写清，本次实现选择。这里记录代码默认值；改变命令行参数会写入 `run_config.json`。模型结构常量在 `model.py`，训练常量在 `main.py`，数据与指标常量分别在 `data.py`、`evaluation.py`。

## 数据与裁剪

| 参数 | 默认值/行为 | 来源 |
| --- | --- | --- |
| 队列规模检查 | 126；`--expected-patients 0` 可关闭，仅用于其他已定义队列 | P/I |
| 论文参考事件数/位点数 | 97 死亡、29 删失、551 位点；不硬编码为标签或伪造输入 | P |
| 生存起点 | 治疗后 PET/CT 检查日 | P |
| 天/月 | 30.4375 | P |
| 生存 duration | 有限且 >0；event∈{0,1} | P/I |
| 病灶纳入 | 人工确认 definite/probable；indeterminate 排除；不按体积自动删除 | P |
| 事件编码 | matched=0、disappeared=1、new=2 | I |
| 部位编码 | primary=0、nodal=1、other=2、unknown=3 | I |
| 位点中心 | 两时间点配准后掩膜联合包围盒中心；单侧存在时用该侧 | P |
| 包围盒坐标约定 | 取掩膜非零体素包围盒中心的物理角点，转换到 LPS 轴向范围 | I |
| PET 裁剪 | 64³、3 mm；192 mm 视野 | P |
| CT 裁剪 | 96³、2 mm；192 mm 视野 | P |
| 裁剪方向 | LPS 单位方向矩阵，数组顺序 z/y/x；共同物理中心 | I |
| 图像插值 | 三线性；不重新执行纵向配准 | P/I |
| PET 归一化 | log1p(clip(SUV,0,25))/log(26) | P |
| CT 归一化 | clip(HU,-1000,1000)/1000 | P |
| 图像边界填充 | PET 原始 SUV=0；CT 原始 HU=-1000 | P |
| 缺失病灶的影像 | 读取该空间位置的实际图像，绝不按事件置零 | P |
| 病灶数量 | 不截断、不抽样；逐位点编码 | I |
| 输入精度 | float32；NPZ 压缩保存 | I |
| 图像范围验证容差 | 1e-5 | I |
| 输入几何检查 | 每位患者必须有 baseline 病灶；位点中心必须在四个实际图像的物理覆盖内（允许半体素边界） | P/I |
| 时间变量 | prepost_interval_days、post_minus_rt_end_days | P |
| 时间缺失 | fitting 中位数填补；全缺失时填 0；附 2 个缺失指示 | P/I |
| 时间标准化 | fitting 有效值均值/总体标准差（ddof=0）；填补后应用 | P/I |
| 标准差下限 | 1e-8；整列缺失时 scale=1 | I |
| 代谢目标顺序 | ΔSUVmax；%ΔSUVmax；%ΔMTV；%ΔTLG | P |
| 百分比定义 | 100×(post-pre)/pre | P/I |
| 目标有效性 | matched、quantitative_reliable=1、有限非负定量值；百分比 pre>0 | P/I |
| 目标标准化 | 仅 fitting 有效病灶的逐列均值/总体标准差；全缺失目标全程禁用 | P/I |
| 原始测量 | SUVmax；MTV(mL)；TLG=SUVmean×MTV，外部原始定量图测量 | P |

## 网络

| 参数 | 默认值/行为 | 来源 |
| --- | --- | --- |
| 模态编码器 | PET/CT 独立；各自 pre/post 共享权重 | P |
| Stem | Conv3d 1→32，kernel=3，stride=2，padding=1，无 bias；GN+SiLU | S/I |
| 四 stage 通道 | 32、64、128、256 | S |
| stage stride | 1、2、2、2 | S |
| 每 stage | 1 个残差 block，2 个 3³ Conv；第一层应用 stage stride | S/I |
| 残差 shortcut | 通道/步幅改变时 1³ Conv+GN，否则 identity | I |
| 卷积 padding/bias | 3³ 卷积 padding=1，1³ padding=0；编码卷积无 bias | I |
| GroupNorm | 8 组，eps=1e-5，affine=True | S/I |
| 卷积激活 | SiLU；第二个 GN 后残差相加，再 SiLU | S/I |
| 全局特征 | 全局均值池化，Linear 256→128，含 bias | P/S/I |
| CT token | 1³ Conv 256→128，含 bias；adaptive average pool 6³；216×128 | P/S/I |
| CT cross-attention | 1 层、4 heads；post Q、pre K/V；dropout=0.2；无额外位置编码/残差 | P/I |
| CT attended pooling | 216 tokens 的均值 | P/I |
| 事件/部位 embedding | 各 8 维，无 padding 类别 | S/I |
| 时间 MLP | 4→32→16 | S |
| PET response MLP | 536→256→128；pre/post/post-pre/post×pre/time/event | P/S |
| PET prototypes | 4×128，可学习；cosine softmax τ=0.1 | P |
| 原型重建 | Σ assignment×原型；分配时 L2 normalize，重建用可学习原型本身 | P/I |
| PET combined MLP | 260→128→128；response/reconstruction/4 assignments | P/S |
| CT response MLP | 536→256→128；attended/pre/post/post-pre/time/event | P/S |
| 模态投影 | PET/CT 各 Linear 128→128，含 bias | P/S/I |
| gate MLP | 408→128→128 后 sigmoid；p/c/abs(p-c)/time/event | P/S |
| fusion MLP | 800→256→128；p/c/p×c/p-c/abs(p-c)/gate融合/type/event/time | P/S |
| Set Transformer | 2 SAB；1 seed 的 PMA；128 维、4 heads | P |
| attention block | MHA 残差+LayerNorm，FFN 残差+LayerNorm；post-norm | I |
| Set FFN | 128→256→128；dropout=0.2 | S |
| PMA | 可学习 seed 作 query；病灶作 K/V；同一 attention block 结构 | P/I |
| LayerNorm | eps=1e-5，affine=True | I |
| 所有 MLP | Linear→SiLU→Dropout(0.2)→Linear；末层不加激活（gate 另 sigmoid） | I |
| MHA | batch_first=True，bias=True，attention dropout=0.2，4×32 维 head | I |
| 原型分布 | 对患者所有位点的 assignment 等权平均 | P |
| 分布熵 | -Σq log(q+1e-8) | P |
| 生存头 | 133→64→5，输出 logits；hazard=sigmoid(logits) | P/S |
| 辅助头 | PET response→128→64→4 | P/S |
| 形态辅助监督 | 无 | P |
| Conv 初始化 | Kaiming normal、fan_in、relu gain；bias=0 | S/I |
| Linear 初始化 | Xavier uniform、gain=1；bias=0 | S/I |
| MHA QKV 初始化 | PyTorch 默认 Xavier uniform，QKV bias=0；输出投影同 Linear | I |
| embedding 初始化 | normal(mean=0,std=0.02) | I |
| GN/LN 初始化 | scale=1、bias=0 | I |
| PMA seed 初始化 | Xavier uniform | I |
| prototype 初值 | 标准正态后 L2 normalize；warm-up 后被 k-means 替换 | I |
| L2 normalize eps | PyTorch 默认 1e-12 | I |
| 参数总数 | 8,482,865（本次实现实测） | I |

## 损失、优化和增强

| 参数 | 默认值/行为 | 来源 |
| --- | --- | --- |
| 生存区间 | fitting 死亡时间 20/40/60/80/100 百分位；b0=0 | P |
| quantile | 线性插值；重复边界明确报错 | S/I |
| 死亡似然 | (b[j-1],b[j]]；之前区间 log(1-h)，事件区间 log(h) | P |
| 删失似然 | 仅 b[j]≤观察时间的已完整观察区间 | P |
| 尾部观察 | 超过最终边界按该边界行政删失；死亡恰好在边界仍算死亡 | P |
| NLL 数值 | softplus(logits) 和 softplus(-event_logit)，FP32 | S/I |
| 总损失 | Lsurv+0.20 Lmet+0.05 Lproto | P |
| Lmet | Smooth L1，beta=1；先各病灶有效目标均值，再患者内病灶均值，再有效患者均值 | P/S |
| 无有效目标批次 | Lmet=0 | S |
| Lproto compactness | mean_i Σk a_ik×(1-cos(r_i,μ_k))；再患者平均 | P/I |
| Lproto diversity | 非对角原型两两 cosine 的 mean(relu(cos-0.2)²) | S/I |
| 两原型项权重 | compactness + diversity，1:1；整体乘 0.05 | I |
| warm-up | 20 epoch，固定初始 LR；uniform assignment=0.25；重建输入=0；无 Lproto | P/S |
| prototype initialization | fitting、无增强、eval 模式 response；L2 normalize；k-means K=4 | P/S |
| k-means | n_init=20、max_iter=300、tol=1e-4、algorithm=lloyd、init=k-means++；归一化中心 | S/I |
| joint 训练上限 | 150 epoch，含完整原型分支 | P |
| 优化器 | AdamW；betas=(0.9,0.999)，eps=1e-8，amsgrad=False | P/S/I |
| LR | PET/CT 编码器 1e-4；其他模块 3e-4 | P |
| weight decay | 1e-4；排除 bias 及一维参数（包括归一化 scale） | P/S/I |
| 其他 decay 范围 | embedding、prototype、PMA seed 等二维/三维参数也正则化 | I |
| cosine schedule | joint 阶段从初始 LR 到其 1%，T=150；每 epoch 后更新 | P/S |
| 有效 batch | 4 患者；1 患者计算并反传一次，累积 4 次 | P/I |
| 与本地补充 batching 差异 | 补充为 2×2；本实现为 1×4，避免多病灶显存堆积；有效 batch 和患者加权保持一致 | I |
| 尾部 batch | 不丢弃，按实际患者数归一；辅助项按本组有效患者数归一 | I |
| 梯度裁剪 | 全局 L2 norm 最大 5.0；非有限梯度报错 | S/I |
| CUDA 精度 | FP16 autocast + GradScaler；CPU FP32；--no-amp 关闭 AMP | S/I |
| GradScaler 默认 | init_scale=65536、growth_factor=2、backoff_factor=0.5、growth_interval=2000 | I |
| 显存控制 | 编码器 non-reentrant activation checkpoint，默认开启 | I |
| checkpoint 选择 | joint 阶段验证 survival NLL 的实际最低值；不把辅助损失用于选择 | P/S |
| 早停 | patience=25；显著改进阈值 1e-4；与最低 checkpoint 单独记录 | P/S/I |
| 几何增强概率 | 每位点 0.5 | S |
| 旋转 | 各轴独立 uniform[-10°,10°]；Rz·Ry·Rx | P/S/I |
| 平移 | 各轴独立 uniform[-5,5] mm | P/S |
| 尺度 | isotropic uniform[0.95,1.05] | P/S |
| 增强物理定义 | 绕共同中心缩放旋转后平移；逆变换采样；四个 crop 共用变换 | P/S/I |
| 增强采样 | grid_sample 三线性，align_corners=False；物理视野 192 mm | I |
| CT 噪声 | 各时间点独立 p=0.2；Gaussian std=10 HU，即归一化 std=0.01，之后 clip[-1,1] | S/I |
| PET 强度增强 | 不使用 | P |
| 验证/测试/k-means 输入 | 无任何增强 | P/S |
| 重复种子 | 42、73、101、137、211 | S |
| 每折训练/内层划分/k-means seed | repeat_seed×100+zero_based_fold | I |
| 数值可重复设置 | Python/NumPy/PyTorch seed；cuDNN benchmark=False、deterministic=True；deterministic algorithms warn_only=True；CUBLAS_WORKSPACE_CONFIG=:4096:8 | I |
| 可重复性边界 | CUDA grid_sample 等可能仍为非确定性；代码不承诺不同驱动/GPU间逐 bit 一致 | I |
| CPU threads/workers | 8；同步逐患者读取，DataLoader workers=0 | I |

## 验证与主评估

| 参数 | 默认值/行为 | 来源 |
| --- | --- | --- |
| 外层 | 5 次重复×5 折；事件分层；shuffle=True | P |
| 内层 | 外层训练患者中 20% 验证；事件分层；余下 fitting | P |
| 模型选择后 refit | 不额外合并 fitting+validation 重训 | P |
| 拟合统计量范围 | timing、targets、事件边界、k-means 仅 fitting | P |
| OOF | 每患者每重复恰好一次，来自该重复的外层测试折 | P |
| 公共预测网格 | 0–36 月，步长 0.25 月 | I |
| 曲线映射 | log-survival 线性插值；S(0)=1；区间恒定 hazard | P/S |
| 重复聚合 | 5 次生存概率算术均值，不平均 logits/hazards | P |
| 支持范围 | 超出任意一次末边界记 NaN，不外推或忽略 NaN 平均 | P |
| 风险 | 1−mean_repeats S(24) | P |
| 主指标 | Uno C 截断 24 月、Harrell C、累计/动态 AUC@24、IBS[0,24] | P |
| C-index 风险 | 统一用聚合 24 月死亡概率 | P |
| 风险 ties | 严格相等算 0.5；无额外浮点 tolerance | I |
| 可比对 | 较早死亡者对更晚观察者；同时间死亡/删失也可比；同时间双死亡不比较 | I |
| Uno 事件截断 | 事件时间≤24；事件权重 1/G(T−)² | P/I |
| AUC case/control | event=1 且 T≤24 为 case；T>24 为 control | P/I |
| AUC 权重 | case 1/G(T−)，control 同时点权重约去 | I |
| IBS 网格 | 0–24 月，每 0.25 月；梯形积分/24 | S |
| Brier IPCW | 死亡 T≤t 用 S(t)²/G(T−)；T>t 用 (1−S(t))²/G(t)；其余=0；除全部患者数 | I |
| 删失估计 | 在聚合 OOF 评估队列上估计反向 KM；死亡/删失同时发生时先移除死亡再估计删失跳变 | I |
| 删失估计范围说明 | 用于评估权重，不用于网络拟合或模型选择；正文未指定删失分布估计范围，本实现不冒称采用原始程序同一约定 | I |
| 无事件/controls/可比对或无 IPCW 支持 | 相应指标为 null；不人为裁剪权重或补结果 | I |
| bootstrap | 2000 患者有放回重采样；seed=20260819 | P/S |
| bootstrap 删失分布 | 每次从抽样患者重新估计 | I |
| CI | 2.5%、97.5% percentile，NumPy 线性 quantile | P/I |
| 无效 bootstrap | 不重抽替换；每指标记录有效/无效次数 | S |
| 不确定性范围 | 条件于固定训练模型和 OOF 预测，不包含重训变异 | P |
| 默认 device | cuda；可明确选 cpu | I |
| 恢复方式 | 只跳过已完成折，未完成折重跑；输入/代码/主要环境指纹必须一致 | I |

## 软件与验证范围

Python 建议 3.12。依赖允许 NumPy [2,3)、PyTorch [2.5,3)、scikit-learn [1.5,2)、SimpleITK [2.4,3)，以兼容运行机器的 CUDA 安装；它们不是论文披露参数。实际运行需保留 `environment.txt`，主要版本也由程序自动记录。这里没有杜撰论文的依赖版本或硬件。

本次使用工作区现有环境进行语法、真实尺寸模型前向/反向、删失边界、代谢目标屏蔽、时间插值和评估数值检查，并通过合成数据单折 1 epoch warm-up + 1 epoch joint 的端到端检查。正式 126 患者训练由使用者运行。人工审核、SUV 转换和纵向配准是数据输入前提，其人工结论不由本主实验代码产生。

本机验证包版本：PyTorch 2.13.0、NumPy 2.5.2、scikit-learn 1.9.0、SimpleITK 2.5.6；CPU 验证。CUDA/AMP 在本机未实测。依赖范围下限不是本次实测版本，也不是论文原始环境声明。
