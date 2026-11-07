# 运行与复核

## 安装

使用Python 3.11及适合设备的PyTorch。对联和赏析推理在本项目CUDA环境验证；原始GPT续写也支持CPU。环境版本记录在 `requirements/environment-lock.txt`。

```bash
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python -m couplet_gpt.evaluation.verify --require-weights
```

安装后也可用 `couplet-gpt` 与 `couplet-gpt-verify`。直接在项目根目录使用 `python -m couplet_gpt` 不要求先安装包。

## 模型与资源

四份权重位于 `models/`，SHA256在 `models/manifest.json`。有规则与无规则共享对联权重，配置分别为 `configs/couplet_rules.json` 和 `configs/couplet_unconstrained.json`。

权重、完整语料、数据库与特征缓存不进入Git。当前没有公开的权重托管地址；克隆代码后应将对应工件复制到清单位置，不能把只有代码的仓库说成开箱可用的完整模型。

对联需要完整ChineseBERT模型目录，文件与固定版本见 [外部模型清单](../models/external/manifest.json)。缺少本地上下文缓存时会现场计算特征。资源与许可见 [第三方声明](../THIRD_PARTY_NOTICES.md)。

## 两项任务

```bash
# 预训练底座续写
python -m couplet_gpt pretraining --text "春风" --max-new 64

# 对联：同一权重的两种解码配置
python -m couplet_gpt couplet --text "春风送暖千山绿" --mode rules
python -m couplet_gpt couplet --text "春风送暖千山绿" --mode unconstrained
python -m couplet_gpt couplet --text "春风送暖千山绿" --mode unconstrained --forward-only

# 完整诗文赏析
python -m couplet_gpt poetry --text "白日依山尽，黄河入海流。欲穷千里目，更上一层楼。" --max-new 512
```

Windows可使用 `launchers/run.ps1`。原有 `scripts/generate.py` 和 `scripts/verify_project.py` 只保留薄包装，正式实现均在包内。

## 评测复核

已有详细候选和分数时，仅重新汇总：

```bash
python -m couplet_gpt.evaluation.ablation --phase summary
python -m couplet_gpt.evaluation.report
```

完整生成／反向评分使用 `--phase all`，需要最终权重、ChineseBERT、词库及相同固定输入。它保留同一搜索预算和冻结反向权重，不会重新训练。完整运行数据位于 `results/couplet/ablation/`，候选、分数和预测仅保留本地。课程结果采用的原始记录也保留在归档中，整理后的指标按语义名称逐项对应，未更改分数。

## 训练代码与历史协议

预训练、对联、反向和赏析代码分别在 `couplet_gpt/training/pretrain.py`、`couplet.py`、`reverse.py`、`appreciation.py`。训练与评测只在显式调用入口时启动，导入包不会自动训练。

完整历史重训还依赖当时的数据划分、初始化权重和冻结协议；对联后期续训的历史初始化工件不等于最终发布权重。这些文件保留在本地归档和兼容路径中，不随Git提交。已整理的训练源代码不能被宣传成仅凭公开目录就能一键复现所有历史阶段；准确的原始源代码快照、依赖和移动映射位于本地 `archive/organization/`。

## 验证与GitHub

```bash
python -m unittest discover -s tests -v
python -m compileall -q couplet_gpt scripts
git status --short
```

默认提交代码、配置、三个阅读文档、最终PDF／LaTeX、来源声明与小型结果汇总。`.gitignore`排除模型大文件、语料、缓存、归档和本地兼容入口。GitHub检查只运行无权重结构测试，不伪装成通过了GPU训练或文学质量测试。

目录收敛、README介绍用途／快速开始／结果／边界，以及将大工件与代码分开，参考GitHub官方的 [README说明](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-readmes) 和 [大文件说明](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)。GitHub没有规定所有ML项目必须使用同一套目录，当前分层是根据本项目两项任务与复现需求作出的选择。

源码仓库为 https://github.com/Glinqo/CoupletGPT 。Git仓库分发代码、配置、说明和报告，大型运行工件按上文位置单独准备。项目尚未指定自有代码许可证；第三方许可独立保留。
