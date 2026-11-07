# 结果与边界

本页是两项任务的统一结果入口。机器可读结果集中在 `results/`；正式课程正文与样例见 [报告](../reports/project_report.pdf) 和 [附录](../reports/sample_outputs.pdf)。

## 预训练与迁移

原始预训练约3000万正文Token，最终加权验证CE约4.7773。训练曲线、训练步日志、验证历史在 [预训练结果](../results/pretraining/)。早期“预训练初始化与随机初始化直接训练对联”的必要对照位于 `results/pretraining/transfer_comparison/`，用于支撑课程作业中的迁移分析；它与后续扩大数据、词库、模型后的结果不是同预算因果对照。

## 对联：最终14组消融

相同2617个锁定上联，7种设置×2种候选选择。所有组冻结同一对前后向权重；每组两种选择共用候选池。下面的chrF是0—100尺度的字符参考相似度。

|推理或组件条件|仅正向chrF|双向chrF|仅正向结构代理|双向结构代理|
|---|---:|---:|---:|---:|
|保留外部规则|3.0424|3.1585|100.00%|100.00%|
|关闭规则、保留完整模型|2.9625|3.0937|76.73%|76.88%|
|再关闭词汇融合|2.5367|2.7158|72.18%|73.02%|
|关闭上下文特征投影|2.5989|2.8462|77.26%|77.88%|
|关闭词组残差向量|2.7005|2.9932|76.88%|77.03%|
|仅匹配旧小词库|2.8783|3.0386|75.54%|75.12%|
|关闭字位对齐|0.5263|0.6045|0.15%|0.73%|

主要发现：

1. **规则提供结构保证。** 关闭规则后，完整模型仍几乎总能等长并自然结束；双向结构代理从100%降至76.88%，缺口主要来自重复模式与同位用字。
2. **反向选择的小幅收益尚不稳定。** 无规则完整模型chrF由2.9625升到3.0937，但逐句chrF差95%配对区间为[−0.0620,+0.3571]，跨零，不能声称稳定优于仅正向。
3. **当前权重中词汇融合有贡献。** 无规则双向条件下，屏蔽后chrF降至2.7158；屏蔽相对完整模型的逐句chrF差区间[−0.5314,−0.0638]未跨零。单项投影、词向量和词库差异的证据弱于关闭整个融合分支。
4. **模型强依赖字位对齐。** 关闭后双向约24%为空输出；这说明当前权重依赖该通道，不能证明另一套删除通道后重训的模型必然失败。

347例固定候选池交叉诊断中，完整评分器在两类候选池中均得到较高双向chrF，但完整词汇生成池并非始终较好。这更支持词汇改善了评分判断的解释，不能把候选数量、候选内容与评分收益混为一谈；候选并集也不是同预算成绩。

### 数据在哪里

|用途|位置|
|---|---|
|全部指标及置信区间|[`metrics.json`](../results/couplet/ablation/metrics.json)|
|规则、长度、停止与失败统计|[`constraint_metrics.json`](../results/couplet/ablation/constraint_metrics.json)|
|固定候选池交叉诊断|[`candidate_scoring_metrics.json`](../results/couplet/ablation/candidate_scoring_metrics.json)|
|固定抽取的原样输出|[`qualitative_examples.jsonl`](../results/couplet/ablation/qualitative_examples.jsonl)|
|逐条预测（本地）|`results/couplet/ablation/predictions/`|
|全部候选（本地）|`results/couplet/ablation/candidates/`|
|前后向训练轨迹与校准|`results/couplet/training/`|

文件名直接表达条件，例如 `unconstrained_forward_only.jsonl`、`without_lexical_fusion_bidirectional.jsonl`；不需要先查R0/R1或历史版本号。

## 诗文赏析：同批底座对照

完整赏析CE使用预先固定的1024条测试样本；自由生成使用其中固定64条，greedy、最多512个新Token。两组使用相同输入与参考，程序字音首句不计入正文指标。

|指标|原始底座|完整赏析微调|
|---|---:|---:|
|正文CE|7.5196|1.5715|
|自然结束率|73.4%|98.4%|
|严重循环率|67.2%|20.3%|
|参考chrF|1.793|20.748|

微调显著改善参考相似度与结束表现，但仍有循环、语义误读和模板化表达，不能视为可靠赏析器。二字锚点或词面重合不能证明理解了原文；赏析参考本身也可能包含AI生成错误。

指标在 [`evaluation_metrics.json`](../results/poetry_appreciation/evaluation_metrics.json)，输出在 [`generated_appreciations.jsonl`](../results/poetry_appreciation/generated_appreciations.jsonl)，对照输出在 `pretrained_outputs.jsonl`。

## 这些结果能说明什么

此次组件实验是固定权重推理干预，不是删除组件后重新训练；置信区间针对固定模型的逐句指标，不能覆盖训练种子差异。测试集历史上使用过，不能称为全新盲测。结构代理、句脚音类和chrF都不能代替专家文学评价。
