（交流可以用英文，所有文档中文）

## 项目目标
编写一份《多模态大模型预训练教程》的中文课程markdown
文件组织是 index.md + chapter1.md + ...
不写代码

###
写一本中文公开教程 markdown讨论如何做一个面向 Vision-Language Action 模型、自动驾驶/具身和用户语音交互场景的多模大模型预训练（支持文字、音频、视频、图像、3D；含代码、表格），目标读者是 ai scientist 和中厂 infra 工程师。从原理到所有重要实操细节。给出1B、10B两档的生产级方案。
数据需要从头搞起（从设计 recipe 到如何爬取 youtube 等细节，共计 30T token（中文/英文 90%，10% 其他语种。token 数要分解到具体数据集）；完全自采（抓取+开源语料，尽量用上别人洗好的数据包）；数据治理要重点讲；部分采用合成数据来补足数据采集困难，如合成音频和文字（类似 Phi-3 的教科书；构成强大的 mid-training；和 agentic RL self-play 产生数据），视频数据量大（需讨论存储与搬运成本）；讨论用 fasttext 过滤和小模型过滤脏数据的方案（其他模态的也要给出））；底下的 infra 是基于 megatron（预计 256x H100 80GB 规模，一次训练以过完 10T token 一遍为准）；模型结构计划选用各模态专用 tokenizer（讨论连续 feature 的兼容方案），然后主体是类似 qwen 的 autoregressive transformer（dense 或 MoE，文字 tokenizer 是 qwen tokenizer，多模扩词表；早期融合）；最后交付的是 pretrain checkpoint；模型训练讨论是否用 Gemma 式 logits 蒸馏；评测完整关心主流指标。要有专门章节讨论项目管理和人员阵型。教程由 index.md + chapter1.md + chapter2.md + ... 组成。要有一个整体时间线贯穿前后。
包含方言/少数语种，以 IPA 兜底；目标分辨率/帧率/码率=6 camera 480p@12Hz；MoE 采取先进架构；驾驶视频含多摄/环视；启用 FP8 主干 + BF16 优化器；3D偏好 blender/CAD 脚本这样的程序格式（如果有可能），然后是 X3D一样的文本结构化格式，其次是 .obj 这种。
章节讨论生成理解一体架构

## 章节结构要求
每个章节应包含：
1. **开篇段落**：简要介绍本章内容和学习目标
2. **文字论述**：以文字论述为主，适当配上公式和 ASCII 图说明。如有数学公式，用 latex. 要有 rule-of-thumb
3. **本章小结**：总结关键概念和公式
4. **常见陷阱与错误** (Gotchas)：每章包含该主题的常见错误和调试技巧
