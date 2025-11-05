# 第 24 章：交付与复现

**[里程碑 W25-W26]**

## 开篇段落

预训练的完成并非项目的终点，而是价值转化的起点。一个凝聚了数百万美元算力成本和数千人·时投入的模型，如果无法被他人稳定使用、精确验证和高效复现，其大部分价值都将被锁定在少数核心开发者的本地环境中。本章聚焦于预训练完成后的“最后一公里”——如何将训练产出的海量、分散的文件（权重、优化器状态、词表、超参数）系统性地打包成一份清晰、可用、可复现的生产级工程交付物。我们将深入探讨分布式 Checkpoint 的内在结构与合并策略，多模态 Tokenizer 的原子化封装，复现脚本的健壮性设计，以及作为负责任 AI 实践核心的模型卡（Model Card）的撰写规范。学完本章，您将能够将一个极端复杂的预训练项目成果，转化为一个可供下游团队、合作伙伴乃至整个社区直接使用的、文档齐全、值得信赖的工程资产。

## 文字论述

### 24.1 Checkpoint 结构与分片

在 Megatron 这样的大规模分布式训练框架下，模型 Checkpoint 远非单个文件，而是一个反映了训练时硬件拓扑和并行策略的复杂目录结构。理解并妥善管理这一结构，是后续一切应用（微调、评估、推理）的基石。

**1. 训练 Checkpoint vs. 推理 Checkpoint：分离关注点**

*   **训练 Checkpoint (Training Checkpoint)**: 这是训练过程的“完整快照”，其首要目标是 **容错与续训**。它体积庞大，通常包含：
    *   **模型权重分片 (Sharded Weights)**: 按照张量并行（TP）和流水线并行（PP）策略切分的模型参数。例如，一个 `nn.Linear` 层的权重矩阵可能按列（对于 TP）被切分到不同 GPU 上。
    *   **优化器状态分片 (Sharded Optimizer States)**: 这是 Checkpoint 体积的主要构成部分。对于 Adam/AdamW 优化器，每个参数都对应一阶矩 (`m`) 和二阶矩 (`v`)，其大小通常是模型参数本身的 **两倍** (BF16/FP32)。这些状态同样按 TP/PP 切分。
    *   **学习率调度器状态 (LR Scheduler State)**: 记录当前的学习率、步数等。
    *   **随机数生成器状态 (RNG State)**: 保证从断点续训时，数据加载、dropout 等随机过程可以精确复现。
    *   **训练元数据 (Metadata)**: 包括迭代步数、已处理 Token 数、损失值历史等。

*   **推理 Checkpoint (Inference Checkpoint)**: 其核心目标是 **易用性与效率**。它只包含模型权重，经过合并与格式转换，通常具备以下特点：
    *   **合并与去冗余**: 所有分片权重被合并，优化器状态等无关信息被丢弃。
    *   **标准格式**: 通常转换为业界标准格式，如 Hugging Face Transformers 的 `.bin` 或更安全的 `.safetensors` 格式，便于生态系统集成。
    *   **体积优化**: 体积通常只有训练 Checkpoint 的 1/3 或更少。

**Rule-of-Thumb**: 永远不要将原始的、高度分片的训练 Checkpoint 直接暴露给下游用户。提供一个专门的、经过验证的转换脚本，将训练 Checkpoint 转换为简洁的推理 Checkpoint，这是交付流程中的一个关键步骤。

**2. 目录结构深度解析**

一个典型的 Megatron-LM Checkpoint 目录结构如下，它编码了并行信息：

```ascii
<checkpoint_root>/
├── iter_0010000/
│   ├── pp_rank_000/
│   │   ├── mp_rank_00/
│   │   │   └── model_optim_rng.pt  <-- TP=0, PP=0 的权重/优化器/RNG
│   │   └── mp_rank_01/
│   │       └── model_optim_rng.pt  <-- TP=1, PP=0 ...
│   ├── pp_rank_001/
│   │   ├── mp_rank_00/
│   │   │   └─ model_optim_rng.pt  <-- TP=0, PP=1 ...
│   │   └── mp_rank_01/
│   │       └── model_optim_rng.pt  <-- TP=1, PP=1 ...
│   ├── ...
│   ├── latest_checkpointed_iteration.txt  <-- 文件内容为 "10000"
│   └── arguments.json                     <-- 保存训练时的所有命令行超参数
└── iter_0020000/
    └── ...
```
*   **专家并行 (MoE) 的复杂性**: 如果启用了 MoE，每个专家的权重会作为独立的参数组存储在对应的 Transformer 层中，同样遵循 TP/PP 的切分规则。这使得 Checkpoint 结构更加复杂。

**3. 合并脚本 (Consolidation Script) 的逻辑**

转换脚本的核心任务是“逆向工程”并行切分的过程：
1.  **加载元数据**: 读取 `arguments.json` 以获知训练时的 `tensor-model-parallel-size` 和 `pipeline-model-parallel-size`。
2.  **迭代加载分片**: 按 `pp_rank` 和 `mp_rank` 的顺序遍历，加载所有 `model_optim_rng.pt` 文件，并仅提取其中的 `model` state_dict。
3.  **拼接张量并行 (TP) 分片**:
    *   对于 **行并行** 线性层（如 FFN 的 `dense_h_to_4h`），将不同 `mp_rank` 的权重沿 **行维度** (dim 0) `torch.cat`。
    *   对于 **列并行** 线性层（如 FFN 的 `dense_4h_to_h`），将不同 `mp_rank` 的权重沿 **列维度** (dim 1) `torch.cat`。
4.  **组装流水线并行 (PP) 分片**: 将不同 `pp_rank` 的层（例如，`pp_rank_000` 包含 0-11 层，`pp_rank_001` 包含 12-23 层）按顺序组装成一个完整的 `state_dict`。
5.  **格式转换与保存**: 将合并后的 `state_dict` 键名映射到目标格式（如 Hugging Face），然后使用 `torch.save` 或 `safetensors.save_file` 保存。

### 24.2 Tokenizer/词表发布与兼容层

模型权重和 Tokenizer 是一个不可分割的 **原子单元**。任何细微的不匹配都会导致灾难性的解码失败。对于我们这个覆盖文本、音频、视频、3D、IPA 的复杂模型，Tokenizer 的交付必须做到万无一失

**1. 原子化的 Tokenizer 文件包**

交付的 Tokenizer 包必须包含所有必要文件，并附带版本信息：

*   **`tokenizer.json`**: 由 `tokenizers` 库生成的核心文件，包含了词表、归一化、预分词、BPE 模型状态等。这是最高效、最完整的表示。
*   **`vocab.json` / `merges.txt`**: 原始的 BPE 词表和合并规则，提供可读性和向后兼容性。
*   **`special_tokens_map.json`**: 定义了框架所需的通用特殊 Token，如 `{"unk_token": "[UNK]", "bos_token": "<s>", ...}`。
*   **`added_tokens.json`**: **极为关键**。明确列出所有为多模态、IPA、3D 脚本等后期添加的特殊 Token 及其 ID。这份文件的版本必须与模型权重严格同步。例如：
    ```json
    {
      "<|image|>": 50257,
      "<|video_start|>": 50258,
      "<|ipa_ə|>": 50259,
      "<|ipa_ʃ|>": 50260,
      "<|blender:cube|>": 50261,
      // ... 几百个类似 token
    }
    ```

**2. `VLAProcessor`：多模态输入的统一接口**

直接让用户处理多模态输入的 Tokenization 是繁琐且极易出错的。我们必须提供一个高层封装的 Python 类，作为用户与模型之间的唯一接口。

```python
# 伪代码示例
class VLAProcessor:
    def __init__(self, text_tokenizer, image_encoder, audio_codec, ...):
        self.text_tokenizer = text_tokenizer
        self.image_encoder = image_encoder # e.g., a VQGAN encoder
        self.audio_codec = audio_codec   # e.g., an EnCodec model
        # ...

    def process_text(self, text):
        return self.text_tokenizer(text, return_tensors="pt")

    def process_video(self, video_frames):
        # video_frames: [num_frames, H, W, C]
        # 1. Patchify/Tubeletize
        # 2. Project through vision encoder to get visual tokens/embeddings
        # 3. Add special tokens for camera position and timestamp
        ...
        return {"input_ids": visual_token_ids, ...}
    
    def process_3d_script(self, script_string):
        # 1. Validate script syntax
        # 2. Tokenize based on predefined script grammar
        # 3. Map to special token IDs
        ...
        return {"input_ids": script_token_ids, ...}

    def __call__(self, inputs: list, padding=True, return_tensors="pt"):
        # Takes a list of mixed-modality inputs, e.g.,
        # ["Describe this driving scene:", PIL.Image.open("..."), "The action is:", Action.TURN_LEFT]
        # 1. Dispatches each item to the correct processor method.
        # 2. Interleaves the resulting token sequences with modality-specific tokens.
        # 3. Pads the final sequence to a uniform length.
        # 4. Returns a dict ready to be passed to model.forward().
        ...
        return batch
```
这个 `VLAProcessor` 类与模型权重一起交付，极大地降低了用户的使用门槛，并从根本上杜绝了预处理不一致的问题。

### 24.3 推理样例与端到端 Demo

代码胜于雄辩。提供可直接运行的示例是验证交付物完整性、展示模型能力并引导用户的最有效方式

**1. 命令行接口 (CLI) Demo (`cli_demo.py`)**
提供一个简单的 CLI，用于快速的单轮交互，便于集成到自动化测试或脚本中。
```bash
python cli_demo.py \
    --model_path /path/to/inference_checkpoint \
    --image_input /path/to/driving_scene.jpg \
    --text_prompt "The image shows a busy intersection. What is the traffic light's color for the ego vehicle?"
```
这个脚本应该能直接输出模型的文本回答。

**2. SDK 风格的推理样例 (`inference_example.py`)**
这是一个更详细的 Python 脚本，展示了如何以编程方式使用 `VLAProcessor` 和模型，是下游开发者集成模型到自己应用中的“入门教程”。它应覆盖多种核心能力，并包含详细注释。

**3. 交互式 Demo (Jupyter Notebook / Gradio)**
对于一个多模态模型，一个交互式 Web UI 是必不可少的。它能让非技术人员（如产品经理、管理层）直观地感受模型的能力和局限。
*   **功能区**: 应包含文件上传（图片/视频/音频）、文本输入框、3D 脚本输入区。
*   **控制区**: 提供解码参数的滑块或输入框（如温度、Top-p、Top-k）。
*   **展示区**: 清晰地展示多模态输出（生成的文本、动作序列、合成的音频等）。
*   **案例库**: 预置一些有代表性的成功和失败案例，引导用户探索。

### 24.4 复现脚本与配置矩阵

科学的严谨性和工程的可维护性要求预训练过程是完全可复现的。

**1. 封装的启动脚本 (`run_pretrain.sh`)**
该脚本应封装所有启动细节，使其只依赖少数几个环境变量。
```bash
#!/bin/bash
# sbatch directives for SLURM scheduler
# SBATCH --job-name=vla-10b-pretrain
# SBATCH --nodes=32
# SBATCH --ntasks-per-node=8
# SBATCH --gres=gpu:8
# ...

# Environment variables
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=6000
export DATA_PATH="/path/to/blended_dataset"
export CHECKPOINT_PATH="/path/to/checkpoints"

# Megatron-LM launch command
torchrun --nproc_per_node 8 --nnodes 32 --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
    pretrain_vla.py \
    --num-layers 48 \
    --hidden-size 4096 \
    --use-fp8 \
    --tensor-model-parallel-size 8 \
    --pipeline-model-parallel-size 16 \
    # ... (数十个其他参数)
```

**2. 环境快照：超越 `requirements.txt`**
`requirements.txt` 只能锁定 Python 包版本，但无法保证底层的 CUDA, cuDNN, NCCL, 编译器版本一致，而这些对性能和数值稳定性至关重要。
*   **金标准**: 提供一个 `Dockerfile` 或 `Singularity` 定义文件，它从一个固定的基础镜像（如 `nvidia/pytorch:24.03-py3`）开始，精确安装所有依赖。这是实现字节级复现的唯一可靠途径。
*   **次优方案**: 提供详尽的环境文档，列出所有关键软件的版本号，并提供构建脚本。

**3. 配置矩阵与实验追踪**
交付物应包含一个 `configs` 目录，存放 1B 和 10B 模型的完整配置文件 (`.yaml` 或 `.json`)。同时，强烈建议使用实验追踪工具（如 Weights & Biases, MLflow）来记录每次运行，并将运行链接附在文档中。这提供了从代码 commit -> 配置 -> 结果的完整可追溯链条。

| 参数 (Parameter)             | 1B Dense (基线)              | 10B MoE (生产)                      |
| ---------------------------- | ---------------------------- | ----------------------------------- |
| **架构**                     |                              |                                     |
| `num-layers`                 | 24                           | 48                                  |
| `hidden-size`                | 2048                         | 4096                                |
| `ffn-hidden-size`            | 8192                         | 14336                               |
| `num-attention-heads`        | 32                           | 32                                  |
| `moe-num-experts`            | N/A                          | 16 (每 2 层一个 MoE)                |
| `moe-top-k`                  | N/A                          | 2                                   |
| **训练**                     |                              |                                     |
| `global-batch-size`          | 4,194,304 tokens             | 8,388,608 tokens                    |
| `lr-decay-style`             | cosine                       | cosine                              |
| `lr`                         | 3.0e-4                       | 1.5e-4                              |
| `min-lr`                     | 3.0e-5                       | 1.5e-5                              |
| `weight-decay`               | 0.1                          | 0.1                                 |
| **并行**                     |                              |                                     |
| `tensor-model-parallel-size` | 4                            | 8                                   |
| `pipeline-model-parallel-size` | 8                          | 16                                  |
| **精度**                     |                              |                                     |
| `fp8-format` (TE)            | E4M3                         | E4M3                                |
| `optimizer-precision`        | BF16                         | BF16                                |

### 24.5 版本与模型卡

**1. 语义化版本控制 (Semantic Versioning)**
对模型 Checkpoint 和 Tokenizer 应用严格的版本管理 (`v1.0.0`)：
*   **主版本号 (Major)**: 模型架构发生不兼容变化（如层数、注意力机制改变），需要修改推理代码。
*   **次版本号 (Minor)**: 模型在相同架构下进行了大规模增量训练或重要微调，能力显著提升，但接口兼容。
*   **修订号 (Patch)**: 修复了 Checkpoint 中的小问题（如转换脚本 bug、模型卡错别字），模型权重不变。

**2. 模型卡 (Model Card)：负责任 AI 的基石**
模型卡是模型的“说明书”，是提升透明、管理风险和促进负责任使用的核心文档。

*   **模型基本信息**:
    *   模型名称与版本：VLA-Driver-10B-MoE v1.0.0
    *   发布日期：2024-10-26
    *   模型类型：多模态自回归 Transformer (Unified Generation & Understanding)
    *   许可证：Apache 2.0
    *   联系方式：[ai-safety@your-company.com]
*   **预期用途 (Intended Use)**:
    *   **直接用途**: 用于研究目的的闭环/开环自动驾驶策略生成；作为驾驶场景的问答与描述系统；多模态交互式助手原型。
    *   **下游用途**: 可作为基础模型进行微调，以适应特定的具身机器人任务或特定地域的驾驶规则。
    *   **禁止使用的场景 (Out-of-Scope)**: **严禁**用于任何没有人类监督的、直接控制物理车辆或机器人的生产环境。严禁用于生成欺骗性内容、进行交通违规行为或侵犯个人隐私。
*   **训练数据 (Training Data)**:
    *   **数据来源与构成**: 简述 Chapter 4 的据配比，强调数据来源的多样性（网页、代码、YouTube、播客、合成数据）及合规性努力（遵守 robots.txt，使用官方 API）。
    *   **数据治理**: 明确声明已采用自动化工具和人工抽样对数据进行 PII 清洗、去毒化和去偏见处理。但需强调，无法保证 100% 消除所有有害或不准确数据。
*   **性能评测 (Evaluation)**:
    *   **评测摘要**: 以表格形式展示在 Chapter 22 中核心基准测试集上的得分，并与 SOTA 模型进行对比。
    *   **局限性 (Limitations)**:
        *   **事实性**: 模型可能产生“幻觉”，生成不符合事实的文本描述或错误的行动指令。
        *   **鲁棒性**: 在训练数据中未充分出现的罕见场景（如极端天气、非典型交通事故、特殊路标）下，模型性能会显著下降。
        *   **偏见**: 由于训练数据主要来自北美和欧洲，模型可能对其他地区的交通规则、驾驶习惯和文化背景理不足，甚至做出错误的判断。
        *   **因果推理**: 模型主要学习相关性而非因果性，可能无法理解复杂的长时程交通博弈。
*   **伦理与安全考量 (Ethical Considerations)**:
    *   **风险评估**: 分析模型被滥用的潜在风险，如生成虚假驾驶视频、用于恶意目的的机器人控制等。
    *   **风险缓释**: 描述为降低风险所做的努力，如在模型中植入安全过滤器、对敏感主题的生成进行限制，以及在指令微调阶段加入安全与伦理准则。
*   **如何引用 (Citation)**: 提供 BibTeX 格式的引用信息，以便学术界引用。

## 本章小结

本章系统性地阐述了将一个复杂的预训练项目转化为一个健壮、易用、可复现的工程交付物的全过程。一个成功的交付不仅仅是上传一堆权重文件，而是一个包含了 **规范化 Checkpoint**、**原子化封装的 Tokenizer**、**多层次的推理样例**、**固化环境的复现脚本**  **详尽透明的模型卡** 的综合性工程软件包。遵循这些最佳实践，不仅能极大化模型的价值和影响力，更是践行负责任 AI 开发、建立社区信任的不可或-

## 常见陷阱与错误 (Gotchas)

1.  **Checkpoint 与并行策略的“幽灵耦合”**: 直接将训练时的分片 Checkpoint 用于不同并行配置（甚至不同硬件）的推理，会导致权重加载错误或更隐蔽的数值错误。**调试技巧**: 建立一个“黄金”评测集。在合并 Checkpoint 后，立即在单 GPU 上用该评测集跑一遍，确保关键指标与分布式评测结果在误差范围内一致。
2.  **Tokenizer 特殊 Token 的“静默”不匹配**: 在微调或新实验中，无意中改变了 `added_tokens.json` 的内容或顺序，导致模型性能断崖式下跌，但代码不报错。**调试技巧**: 为 Tokenizer 包本身计算一个哈希值（checksum），并在加载模型时校验。强制要求任何对 Tokenizer 的修改都必须触发模型的新版本发布。
3.  **环境依赖的“沼泽”**: 复现脚本在另一台机器上因某个未声明的系统库（如 `libGL.so.1`）或 NCCL 版本不兼容而失败，排查耗时数天。**调试技巧**: 坚持使用 Docker/Singularity 作为交付的唯一环境标准。在 CI/CD 流程中加入一个步骤，自动在一个干净的环境中基于 Dockerfile 构建镜像并运行测试，确保环境的自包含性。
4.  **推理与训练预处理的“像素级”差异**: 推理代码中图像归一化的均值/标准差与训练时有微小差异（如 `0.5, 0.5, 0.5` vs. ImageNet 的均值），导致性能下降。**调试技巧**: 将所有预处理变换（transforms）作为配置参数保存在模型配置 `config.json` 中，并让 `VLAProcessor` 从此配置中读取参数来初始化，确保来源唯一。
5.  **模型卡的“免责声明化”**: 将模型卡写成一份充满了法律术语、旨在规避责任的文档，而不是一份真诚帮助用户理解模型、促安全使用的指南。**调试技巧**: 邀请潜在用户（特别是那些持批评态度的用户）来审阅模型卡草稿。如果他们读完后感到困惑或觉得信息不透明，那就说明模型卡需要重写。包含具体的失败案例截图远比抽象的文字描述更有说服力。
