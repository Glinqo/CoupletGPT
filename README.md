# CoupletGPT

**从零预训练字符级GPT，分别完成中文对联生成与古诗文赏析。** 本项目是《具身智能基础》课程作业：预训练底座由自己实现并训练；对联分支结合冻结ChineseBERT与词汇适配器，赏析分支直接使用完整诗文赏析数据微调。

## 先看什么

1. [方法与代码导读](docs/methodology.md)：做了什么、两项任务如何实现、借鉴了哪些论文。
2. [实验结果与局限](docs/results.md)：最终消融、赏析对照、哪些提升有证据。
3. [运行与复核](docs/reproduction.md)：安装、工件准备、运行命令与复现范围。

正式材料提供最新的 [16页研究报告](reports/project_report.pdf)（含两页附录）和 [独立生成样例附录](reports/sample_outputs.pdf)。LaTeX排版材料集中在 `reports/latex/`，也提供 [LaTeX压缩包](reports/latex_source.zip)。

## 两项任务

|任务|输入与输出|实现|
|---|---|---|
|对联生成|输入上联，生成下联|GPT＋ChineseBERT词汇适配器；可选规则与反向选择|
|诗文赏析|输入完整诗文，输出字音说明与赏析|程序给出字音依据，独立微调的GPT生成正文|

两项任务共享预训练来源，最终使用各自权重。ChineseBERT是冻结外部编码器，不计入从零训练的GPT参数量。

## 快速开始

准备Python 3.11、适合设备的PyTorch与清单中的本地权重，再运行：

```bash
git clone https://github.com/Glinqo/CoupletGPT.git
cd CoupletGPT
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python -m couplet_gpt.evaluation.verify --require-weights

python -m couplet_gpt couplet --text "春风送暖千山绿" --mode unconstrained
python -m couplet_gpt couplet --text "春风送暖千山绿" --mode rules
python -m couplet_gpt poetry --text "白日依山尽，黄河入海流。欲穷千里目，更上一层楼。" --max-new 512
```

对联两种模式共用同一对前后向权重，`--forward-only`可关闭反向选择。`pretraining --text "春风" --max-new 64`可运行原始底座续写。完整命令见 [运行说明](docs/reproduction.md)。

权重、完整数据和缓存保留本地，不放进Git；当前没有公开权重下载地址。工件位置与SHA256见 [模型清单](models/manifest.json)，ChineseBERT版本见 [外部模型清单](models/external/manifest.json)。只有代码的克隆需要补齐这些工件后才能推理。

## 目录

```text
couplet-gpt/
├── couplet_gpt/       # 唯一的正式代码包
│   ├── modeling/     # GPT、字位对齐、词汇适配器、ChineseBERT
│   ├── data/         # 数据构建、字符编码、上下文缓存
│   ├── training/     # 预训练、对联／反向、赏析微调
│   ├── inference/    # 统一入口、搜索、约束、双向选择
│   ├── evaluation/   # 指标、消融、审计与汇总
│   └── resources/    # 字音、词典和规则资源
├── configs/          # 按任务命名的配置
├── models/           # 三类最终模型；权重留本地
├── data/             # 最终数据、词库与特征缓存
├── results/          # 运行记录，按三条任务线归并
│   ├── pretraining/
│   ├── couplet/
│   └── poetry_appreciation/
├── docs/             # 方法、结果、复现，三篇主文档
├── reports/          # 一份报告、一份附录及LaTeX源码
└── tests/            # 不需要权重的仓库检查
```

`scripts/`只有两个正式包装入口。旧版本、中间工件和原始代码在本地 `archive/`；旧路径兼容文件不纳入Git。阅读项目无需先理解历史版本编号。

## 主要结果

对联在相同2617个上联上完成14组固定权重消融：

|条件|chrF|结构代理达标率|
|---|---:|---:|
|有规则＋双向选择|3.1585|100.00%|
|无规则＋双向选择|3.0937|76.88%|
|无规则＋仅正向选择|2.9625|76.73%|
|无规则＋关闭词汇融合＋双向选择|2.7158|73.02%|

规则主要保证结构；词汇融合在本轮干预中有贡献；完整模型中反向选择的小幅收益尚无稳定统计证据。结构代理和chrF都不等于文学合格率。

赏析模型在798397条完整样本上微调，固定生成测试严重循环率由67.2%降至20.3%，仍存在语义误读与循环。数据划分、指标口径与全部限制集中在 [结果说明](docs/results.md)。

## 检查与来源

```bash
python -m unittest discover -s tests -v
```

GitHub Actions只运行无权重检查，GPU推理在本地验证。第三方模型、数据和许可见 [来源声明](THIRD_PARTY_NOTICES.md)。本仓库公开提供项目源码；自有代码尚未指定开源许可证，第三方许可独立保留。
