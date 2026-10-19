# AMMR-Surv 主实验

仅实现完整 AMMR-Surv：已审核纵向病灶输入、成对裁剪、训练、5 次 × 5 折患者级交叉验证、OOF 生存曲线聚合及 4 项主指标。不包含对照模型、消融、敏感性分析、解释性分析或论文绘图。所有 Python 源码均无注释、无 docstring。

依据用户提供的 `bioengineering-4622041_revised.pdf` 和源文件 `main.tex` 的最终方法实现；附件没有补充文件。网络细节参考本项目原有 `AMMR_revision_20261014/reference/AMMR_Surv_Supplementary_Methods.tex` 中与正文一致的设置。该补充文件作为实现参考，不据此宣称恢复了论文原始训练代码。所有参数、来源和本次选择见 [PARAMETERS.md](PARAMETERS.md)。

## 输入准备

使用已经完成 SUV 转换、专家病灶审核、纵向配准及位点匹配的 NIfTI。预处理前置工作依赖人工判读，不在本主实验程序中自动替代。原始 DICOM、自动 GLOW 掩膜或只有患者结局的表格不能直接启动训练。

1. 复制 `patients.template.csv` 为 `patients.csv`，一位患者一行。`landmark_os_days` 是从治疗后 PET/CT 检查日起到死亡或末次随访的天数，必须大于 0；`os_event` 为死亡 1、删失 0。两个时间间隔单位为天，缺失留空。现有 `processed_longitudinal_lesions/manifests/core_126.csv` 含这些列，可直接用作 `--patients`，但仍需确认它对应最终入组患者。
2. 复制 `sites.template.csv` 为 `sites.csv`，一个最终纵向位点一行。`patient_id` 必须属于患者表，`site_id` 在患者内唯一。拆分或合并病灶应先按专家意见形成同一纵向位点的联合掩膜。
3. `event` 只能是 `matched`、`disappeared`、`new`；`anatomy` 只能是 `primary`、`nodal`、`other`、`unknown`。`reviewed=1` 表示已完成恶性判定、配准与匹配质量审核；只纳入确定或很可能恶性的病灶。不要仅为通过程序检查而填写此字段。
4. `pre_pet/post_pet` 是 SUV 单位的 PET；`pre_ct/post_ct` 是 HU 单位的 CT。所有图像和掩膜必须已经映射到治疗后同一物理参考空间，保留正确 NIfTI 几何信息；允许不同体素大小。程序按共同的 LPS 轴向物理坐标裁剪。图像路径可为绝对路径，也可相对于 `sites.csv`。
5. `pre_mask/post_mask` 是该位点的二值掩膜，非零为病灶。matched 两者非空，disappeared 仅 pre 非空，new 仅 post 非空；不存在的一侧路径留空或使用全零掩膜。四个图像路径在三种事件中都必填，绝不以空白图像代替某个时间点。
6. `quantitative_reliable=1` 仅用于测量可信的 matched 位点。六项定量值来自原始定量 PET 和审核后的原始掩膜：SUVmax、MTV（mL）、TLG（SUVmean × MTV）。未测量项留空，不能从归一化网络输入回算。零基线的百分比目标自动屏蔽；new/disappeared 不生成代谢变化辅助标签。

裁剪程序输出每患者一个 NPZ 和 `manifest.json`，含全部病灶，不抽取或截断病灶数。NPZ 内 PET 为 `[N,2,1,64,64,64]`，CT 为 `[N,2,1,96,96,96]`，时间顺序为 pre/post。程序会校验形状和归一化范围，不重复归一化。

## 运行命令

在 Linux NVIDIA GPU 环境或可运行 PyTorch 的环境中，进入本目录。建议 Python 3.12。先安装与机器驱动兼容的 PyTorch CUDA 构建，再执行依赖安装；下面的 `requirements.txt` 不强制特定 CUDA 版本。

```bash
cd /path/to/ammr_main
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c 'import torch; print(torch.__version__); print(torch.cuda.is_available())'
python main.py prepare --patients /path/to/patients.csv --sites /path/to/sites.csv --output data/crops
python main.py train --manifest data/crops/manifest.json --output runs/ammr_main
python -m pip freeze > runs/ammr_main/environment.txt
```

默认完整执行 25 个模型，每模型 20 个 warm-up epoch，随后最多 150 个 joint epoch，验证早停。CUDA 不可用时会明确报错；`--device cpu` 可以运行，但完整实验宜使用 GPU。没有实际测量本队列显存与耗时，不给出未经实测的硬件保证。

中断后继续：

```bash
python main.py train --manifest data/crops/manifest.json --output runs/ammr_main --resume
```

`--resume` 跳过已完整导出预测的折，未完成的折从同一随机种子重新训练；不恢复 epoch 中间状态。代码、输入文件、运行参数和主要依赖版本通过指纹校验，修改后应使用新输出目录。`prepare` 不覆盖已有 manifest。

## 输出

| 文件 | 内容 |
| --- | --- |
| `run_config.json` | 命令参数、依赖版本、种子和代码/输入 SHA256 |
| `repeat_R/fold_F/split.json` | fitting/validation/testing 患者、折内统计量、生存边界 |
| `repeat_R/fold_F/training.jsonl` | 逐 epoch 训练损失、验证 NLL、学习率 |
| `repeat_R/fold_F/best.pt` | 验证生存 NLL 最低模型和折内转换参数 |
| `repeat_R/fold_F/predictions.npz` | 外层测试预测、logits、该折边界 |
| `oof_predictions.npz` | 5 次 OOF 曲线及生存概率平均值，公共 0–36 月网格 |
| `oof_risk24.csv` | 患者级 24 月 OOF 死亡概率及真实结局 |
| `metrics.json` | Uno C@24、Harrell C、AUC@24、IBS 0–24 月及 bootstrap CI |

超过某次预测末边界的时间点记为 NaN，聚合时只要任一次不支持就仍为 NaN，不用较少重复次数平均。任一折无法支持 24 月或事件分位数边界重复时明确报错，不外推、不静默更改方法。

主指标的 IPCW 采用评估队列反向 Kaplan–Meier 删失分布；每个 bootstrap 样本重新估计。此处是正文未明确的评估约定，详见参数表。bootstrap 只重采样固定 OOF 预测，不重新训练。

## 上传 GitHub

代码已整理为独立目录，数据、权重和结果默认被 `.gitignore` 排除。可在运行前或运行后由你上传。

```bash
git init
git add .gitignore requirements.txt README.md PARAMETERS.md main.py model.py data.py evaluation.py patients.template.csv sites.template.csv
git commit -m "Implement AMMR-Surv main experiment"
git branch -M main
git remote add origin https://github.com/YOUR_ACCOUNT/YOUR_REPOSITORY.git
git push -u origin main
```

先在 GitHub 创建空仓库并替换仓库地址。此交付没有替你创建或推送远程仓库，也没有运行真实队列训练。已通过合成 NIfTI 裁剪、真实输入尺寸前向/反向、单折 1 epoch warm-up + 1 epoch joint 端到端检查、病灶置换不变性以及独立 AUC/IBS 数值检查。这些只验证实现可运行，不证明能得到论文中的 0.735。
