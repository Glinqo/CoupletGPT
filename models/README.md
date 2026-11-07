# Final model artifacts

|角色|权重|配置／字符表|
|---|---|---|
|预训练底座|`pretraining/model.pt`|`pretraining/config.json`及项目`tokenizer/`|
|对联前向与反向|`couplet/forward.pt`、`couplet/reverse.pt`|项目`configs/couplet_rules.json`、`configs/couplet_unconstrained.json`|
|完整诗文赏析|`poetry_appreciation/model.pt`|同目录`config.json`、`vocab.json`|

三类任务模型共四份权重，SHA256见 `manifest.json`。两种对联推理模式共享权重；ChineseBERT独立列于`external/`。模型、数据与实验解释统一见项目根目录README和docs，旧模型仅在本地archive中保留。
