# Final experiment results

运行结果按三条任务线收敛。指标解释统一见 [结果说明](../docs/results.md)。

|任务线|主要文件|用途|
|---|---|---|
|`pretraining/`|`training_summary.json`、`training_steps.jsonl`、`validation_history.jsonl`、`loss_curves.png`|底座预训练及迁移对照|
|`couplet/training/`|`forward/`、`reverse/`下的训练摘要与轨迹|最终对联前后向训练、词库与反向架构对照|
|`couplet/ablation/`|`metrics.json`、`constraint_metrics.json`、`candidate_scoring_metrics.json`|末轮14组消融与固定候选池诊断|
|`poetry_appreciation/`|`evaluation_metrics.json`、`generated_appreciations.jsonl`、`pretrained_outputs.jsonl`|完整赏析微调与同批底座对照|

详细候选、逐条预测、反向分数及运行协议在本地保留。无规则输出文件使用 `unconstrained_forward_only`、`unconstrained_bidirectional` 等语义命名；组件消融使用 `without_lexical_fusion` 等名称。原始版本命名及原始审计记录保留在本地归档，整理不改变分数。
