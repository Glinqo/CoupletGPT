# Frozen external encoder

ChineseBERT-base 是对联词汇特征的冻结编码器，不是我们从零训练的模型。仓库：ShannonAI/ChineseBERT-base；固定 revision：aa8b6fa9c3427f77b0911b07ab35f2b1b8bf248b。

将上游完整模型放到 `models/external/ChineseBERT-base/`，包含 `pytorch_model.bin`、`config.json`、`vocab.txt`、`config/pinyin_map.json`、`config/pinyin2tensor.json`、`config/id2pinyin.json`。字形张量来自原始权重。下载 URL 与文件校验值见 `manifest.json`。不重新下载或替换现有本地权重。
