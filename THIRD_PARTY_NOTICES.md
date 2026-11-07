# Third-party notices

本项目包含第三方模型与数据派生资源；来源、许可与自建GPT分别说明，不将其表述为从零训练成果。

|资源|来源|本地说明|
|---|---|---|
|ChineseBERT-base|ShannonAI/ChineseBERT-base|`models/external/manifest.json` 固定版本与下载校验；权重不进Git|
|中文诗词|chinese-poetry/chinese-poetry|`docs/sources/poetry-LICENSE` 与 README|
|古文|原数据构建记录所列开源古文资源|`docs/sources/classical-LICENSE`、来源JSON|
|对联|原对联数据及扩充来源|`docs/sources/couplet-LICENSE`、数据统计与最终报告|
|平水韵|上游平水韵字表|`data/resources/prosody/LICENSE` 和 manifest|
|字音|pypinyin|`data/resources/prosody/pypinyin-LICENSE.txt`|
|词汇语义资源|THUNLP/OpenHowNet|`data/resources/lexical_grammar/LICENSE`，上游MIT声明保留|
|诗文赏析|ChinesePoetryLibrary及处理清单中的来源|完整语料留本地；赏析可能为AI生成，许可与质量不因微调自动获得保证|

最终202585词词库的各来源与保留数量见 `data/lexicon/source_counts.json`，原始下载、许可与构建记录保留在本地归档。LEBERT、ChineseBERT及COLING等论文的借鉴关系见reports/project_report.pdf的参考文献和方法章节。

未为本项目自有代码擅自选择MIT或其他许可证；发布前可单独确定。
