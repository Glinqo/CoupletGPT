中文对联生成研究报告——可修改 LaTeX 材料

主报告：main.tex；独立两页样例附录：supplement.tex。
使用 XeLaTeX 各编译两次，不需要 BibTeX：
  xelatex -interaction=nonstopmode -halt-on-error main.tex
  xelatex -interaction=nonstopmode -halt-on-error main.tex
将 main.tex 替换为 supplement.tex 可单独编译附录。

sections 为正文与两页附录，bibliography.tex 为实际文献，references.bib 为可选书目数据。
figures 为可编辑 TikZ 神经网络图与 PGFPlots 图表，CSV 为作图数值。
默认使用 TeX Live 自带 Fandol 字体；Windows 上以 SimSun 补充少见汉字。
引用为数字上标，标题层级、页眉和 PDF 元数据使用正式报告名称。
材料不含模型实现、训练源码、权重或内部实验文件。

本轮修改：第四章按方法、结果、讨论重组；补充原框架预训练后直接微调的统一输入评测；局限与未来工作压缩，附录保持两页。主报告16页。
