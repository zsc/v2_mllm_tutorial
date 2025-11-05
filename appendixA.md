# 附录 A：配置与脚本模板（可复制）

本附录提供了整个预训练流程中关键环节的可复制、可修改的配置与脚本模板。这些模板旨在作为生产实践的起点，请根据您的具体硬件环境、数据路径和实验需求进行调整。我们的目标是提供“开箱即用”的蓝图，最大限度地减少从理论到实践的摩擦。

---

## A.1 数据管道：抓取/索引/过滤/去重脚本骨架

真实世界的数据管道是模块化、可观测且容错的。以下模板展示了针对不同模态的、更精细的脚本骨架和核心逻辑伪代码，并引入了配置文件管理的最佳实践。

### A.1.1 配置文件 (`config.env`)

将所有路径和关键参数集中管理，便于不同环境（开发、暂存、生产）的切换。

```ini
# --- 全局路径配置 ---
BASE_RAW_DATA_S3="s3://my-company-raw-data"
BASE_PROCESSED_DATA_S3="s3://my-company-processed-data"
LOCAL_TEMP_DIR="/data/temp_processing"
LOCAL_LOG_DIR="/var/logs/data-pipelines"

# --- 模型与工具路径 ---
FASTTEXT_LID_MODEL="/models/fasttext/lid.176.bin"
TEXT_QUALITY_MODEL="/models/classifiers/text-quality-v1.2"
IMAGE_SAFETY_MODEL="/models/classifiers/image-safety-v2.0" # NSFW/水印/徽标
AUDIO_VAD_MODEL="/models/vad/silero-vad.onnx"
ASR_WHISPER_MODEL="large-v3"

# --- 管道参数 ---
DEFAULT_NUM_WORKERS=256
WEB_SCRAPE_USER_AGENT="MyCompanyResearchCrawler/1.0 (+http://my.company/bot.html)"
YOUTUBE_API_KEY="AIzaSyXXXXXXXXXXXXXXXXXXXX" # 严格使用官方API

# --- 视频处理标准 ---
VIDEO_TARGET_RESOLUTION="854x480" # 接近 480p, 16:9
VIDEO_TARGET_FPS=12
VIDEO_TARGET_CRF=28 # H.264 压缩质量, 数值越高压缩率高

```

### A.1.2 文本处理管道 (`run_text_pipeline.sh`)

```bash
#!/bin/bash
set -e
source config.env # 加载配置

# --- 输入参数 ---
SOURCE_NAME=$1 # e.g., "common_crawl_2023_10"
INPUT_S3_PATH="${BASE_RAW_DATA_S3}/text/${SOURCE_NAME}.tar.gz"
OUTPUT_S3_PATH="${BASE_PROCESSED_DATA_S3}/text/${SOURCE_NAME}"
PIPELINE_LOG_FILE="${LOCAL_LOG_DIR}/${SOURCE_NAME}_$(date +%Y%m%d).log"

exec > >(tee -a "${PIPELINE_LOG_FILE}") 2>&1

echo "--- [$(date)] STAGE 1: 拉取与解压 ---"
aws s3 cp ${INPUT_S3_PATH} ${LOCAL_TEMP_DIR}/
# ... 解压逻辑 ...

echo "--- [$(date)] STAGE 2: 粗筛与格式化 (并行) ---"
# process_warc.py: 提取主内容, HTML标签清洗, fastText语言识别
find ${LOCAL_TEMP_DIR}/raw -name "*.warc.gz" | parallel -j ${DEFAULT_NUM_WORKERS} \
    python scripts/process_warc.py --input {} ...

echo "--- [$(date)] STAGE 3: 精细过滤与质量打分 (并行) ---"
# filter_quality.py: 使用小模型进行毒性/广告/低质分类, PII检测
find ${LOCAL_TEMP_DIR}/jsonl -name "*.jsonl" | parallel -j ${DEFAULT_NUM_WORKERS} \
    python scripts/filter_quality.py --input {} --quality_model ${TEXT_QUALITY_MODEL} ...

echo "--- [$(date)] STAGE 4: 全局去重 (Spark/Ray) ---"
# 实际生产中, 这一步通常在分布式计算框架中完成
spark-submit \
  --master yarn \
  --num-executors 512 \
  jobs/deduplicate_text.py \
  --input_path "${LOCAL_TEMP_DIR}/filtered/" \
  --output_path "${LOCAL_TEMP_DIR}/deduplicated/" \
  --dedup_method "minhash_lsh"

echo "--- [$(date)] STAGE 5: WebDataset 打包与上传 ---"
webdataset create --input "${LOCAL_TEMP_DIR}/deduplicated/" ...
aws s3 sync ${LOCAL_TEMP_DIR}/webdataset/ ${OUTPUT_S3_PATH}/

echo "--- [$(date)] 管道 ${SOURCE_NAME} 执行完毕 ---"
```

### A.1.3 视频处理管道核心逻辑 (`process_video.py` 伪代码)

视频处理涉及复杂的时序和多流操作，FFmpeg 是核心工具。

```python
import ffmpeg
import whisper
import json

def process_single_video(video_path, output_dir):
    # 1. 规格标准化: 降采样调整分辨率、重新编码
    (
        ffmpeg
        .input(video_path)
        .filter('fps', fps=VIDEO_TARGET_FPS)
        .filter('scale', size=VIDEO_TARGET_RESOLUTION)
        .output(f"{output_dir}/processed.mp4", vcodec='libx264', crf=VIDEO_TARGET_CRF, preset='fast')
        .run(capture_stdout=True, capture_stderr=True)
    )

    # 2. 提取音频
    ffmpeg.input(f"{output_dir}/processed.mp4").output(f"{output_dir}/audio.wav", acodec='pcm_s16le', ac=1, ar='16000').run()

    # 3. VAD (语音活动检测) 切分
    # vad_segments = apply_vad(f"{output_dir}/audio.wav", model=AUDIO_VAD_MODEL)

    # 4. ASR 转写 (使用 Whisper 获取带时间戳的文本)
    asr_model = whisper.load_model(ASR_WHISPER_MODEL)
    asr_result = asr_model.transcribe(f"{output_dir}/audio.wav", language='zh') # 指定语种

    # 5. 帧提取与过滤
    # - 提取关键帧 (I-frames) 或按固定间隔采样
    # - 使用图像安全模型过滤每一帧
    # safe_frame_indices = filter_frames(video_path, model=IMAGE_SAFETY_MODEL)
    
    # 6. 生成元数据文件
    metadata = {
        'original_path': video_path,
        'duration': ffmpeg.probe(video_path)['format']['duration'],
        'dimensions': (width, height),
        'asr_transcript': asr_result['text'],
        'asr_segments': asr_result['segments'], # 带时间戳
        # 'safe_frame_indices': safe_frame_indices,
        # 'vad_segments': vad_segments
    }
    with open(f"{output_dir}/metadata.json", "w") as f:
        json.dump(metadata, f)

    # 最终打包时, 会将 processed.mp4 和 metadata.json 一起放入 tar 包
```

---

## A.2 Tokenizer 训练与扩表配置

### A.2.1 训练脚本 (`train_tokenizer.py`)

展示如何使用 `tokenizers` 库从原始文本文件训练一个 BPE tokenizer。

```python
import glob
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.normalizers import NFC, Sequence, Lowercase
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

# 1. 定义词表和特殊 Tokens
VOCAB_SIZE = 151936 # 参考 Qwen-1.5-14B
SPECIAL_TOKENS = [
    # --- 基础控制 Tokens ---
    "<|endoftext|>", "<|im_start|>", "<|im_end|>", "[PAD]", "[UNK]",
    # --- 多模态与 VLA 专用 Tokens (新增) ---
    "<|vision_start|>", "<|vision_end|>",
    "<|audio_start|>", "<|audio_end|>",
    "<|3d_start|>", "<|3d_end|>",
    "<|action_start|>", "<|action_end|>",
    # --- 驾驶场景: 相机位姿 ---
    "<|cam_front|>", "<|cam_front_left|>", "<|cam_front_right|>",
    "<|cam_back|>", "<|cam_left_repeater|>", "<|cam_right_repeater|>",
    # --- IPA 与 程序化 3D Tokens (高频部分) ---
    "ə", "ʃ", "ŋ", "θ", "ð", # IPA
    "bpy.ops.mesh", "location=", "rotation_euler=", "scale=", # Blender
]


# 2. 初始化 Tokenizer
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.normalizer = Sequence([NFC(), Lowercase()]) # 可选
tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
tokenizer.decoder = ByteLevelDecoder()

# 3. 配置训练器
trainer = BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=SPECIAL_TOKENS, min_frequency=2)

# 4. 获取训练文件列表
# 假设我们有一个包含多种语言和代码的混合语料
files = glob.glob("/path/to/text/corpus/*.txt")

# 5. 启动训练
tokenizer.train(files, trainer)

# 6. 保存 Tokenizer
tokenizer.save("vla_multimodal_tokenizer.json")
print(f"Tokenizer trained with vocab size: {tokenizer.get_vocab_size()}")

# 7. 测试
encoded = tokenizer.encode("你好世界, Hello world! Let's test IPA: /nɪˈhaʊ/ and bpy.ops.mesh!")
print("Encoded IDs:", encoded.ids)
print("Decoded text:", tokenizer.decode(encoded.ids))
```

### A.2.2 3D Tokenization 策略对比

| 策略 | 优点 | 缺点 | 适用场景 |
| :--- | :--- | :--- | :--- |
| **程序化脚本 (Blender/CAD)** | **极高压缩率**；**语义丰富**，可编辑，可推理；版本化友好 | 需要模型理解语法和执行逻辑；数据源较少 | 需要生成可执行/可辑 3D 内容的任务 |
| **文本结构化 (X3D/XML)** | 结构清晰，层级关系明确；比网格文件可读性高 | 冗余度较高；对复杂拓扑描述能力有限 | 场景描述、简单几何体组合 |
| **传统网格格式 (.obj)** | 简单直接，兼容性好；数据源广泛 | **信息密度低**，大量重复的 `v`, `vt`, `f`；无语义信息 | 作为兜底方案，处理无法程序化的存量数据 |

**程序化脚本 Tokenization 示例:**

原始 Blender Python 代码:
`bpy.ops.mesh.primitive_cube_add(size=2, location=(1, 2, 3))`

Tokenized 序列 (使用扩充词表的 tokenizer):
`[<bpy.ops.mesh>, .primitive_cube_add, (, size, =, 2, ,, <location=>, (, 1, ,, 2, ,, 3, ), )]`
注意 `location=` 被合并为一个 token，这比逐字符 tokenization 高效得多。

---

## A.3 Megatron 启动参数矩阵

### A.3.1 1B Dense 模型启动脚本 (`run_pretrain_1b_dense.sh`)

此脚本为快速迭代和验证端到端流程的基线。

```bash
#!/bin/bash
# 适用于 4-8 个节点 (32-64 GPUs)

# --- 集群与环境 ---
NNODES=4
GPUS_PER_NODE=8
WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))

# --- 路径配置 ---
DATA_PATH="/data/blended_multimodal_data_small/my_dataset_idx"
CHECKPOINT_PATH="/checkpoints/vla-1b-dense"
TOKENIZER_PATH="/models/tokenizer_vla"

# --- Megatron-LM 分布式参数 ---
TP=2  # 节点内 NVLink
PP=4  # 跨节点
# DP = 32 / (2 * 4) = 4

# --- 1B Dense 模型结构 ---
NUM_LAYERS=24
HIDDEN_SIZE=2048
FFN_HIDDEN_SIZE=5632 # SwiGLU FFN 通常是 2/3 * 4 * H
NUM_ATTN_HEADS=16
SEQ_LENGTH=4096

# --- 训练超参数 ---
MICRO_BATCH_SIZE=4
GLOBAL_BATCH_SIZE=1024 # 4 * 4 * 64 = 1024
# Grad Accum Steps = 1024 / (4 * 4) = 64
TRAIN_TOKENS=1000000000000 # 1T tokens (用于基线)
LR=3e-4
MIN_LR=3e-5
LR_DECAY_STYLE="cosine"
LR_WARMUP_TOKENS=375000000 # 375B
WEIGHT_DECAY=0.1
GRAD_CLIP=1.0

# --- 精度配置 ---
FP8_FORMAT="hybrid"
USE_FP8="--fp8-hybrid"

# --- 构建启动命令 ---
torchrun ... pretrain_gpt.py \
  --tensor-model-parallel-size $TP \
  --pipeline-model-parallel-size $PP \
  \
  --num-layers $NUM_LAYERS \
  --hidden-size $HIDDEN_SIZE \
  --ffn-hidden-size $FFN_HIDDEN_SIZE \
  --num-attention-heads $NUM_ATTN_HEADS \
  --seq-length $SEQ_LENGTH \
  --swiglu --use-rotary-position-embeddings --normalization RMSNorm \
  \
  --micro-batch-size $MICRO_BATCH_SIZE \
  --global-batch-size $GLOBAL_BATCH_SIZE \
  --train-tokens $TRAIN_TOKENS \
  \
  $USE_FP8 --bf16 \
  --optimizer adam --adam-beta1 0.9 --adam-beta2 0.95 \
  # ... 其他参数与 10B 脚本类似 ...
```

### A.3.2 10B MoE 模型启动脚本 (`run_pretrain_10b_moe.sh`)

此脚本为生产级训练方案，细节更丰富。

```bash
#!/bin/bash

# --- 集群与环境配置 ---
NNODES=32
GPUS_PER_NODE=8
WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES)) # 256
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000

# --- 数据与模型路径 ---
DATA_PATH="/data/blended_multimodal_data/my_dataset_idx"
CHECKPOINT_PATH="/checkpoints/vla-10b-moe"
TOKENIZER_PATH="/models/tokenizer_vla/tokenizer.model"
WANDB_PROJECT="vla-10b-pretrain-run"

# --- Megatron-LM 分布式参数 ---
TP=4  # 节点内 4-way Tensor Parallel
PP=8  # 跨 8 个节点 Pipeline Parallel
# DP = 256 / (4 * 8) = 8

# --- 10B MoE 模型结构参数 ---
NUM_LAYERS=40
HIDDEN_SIZE=4096
FFN_HIDDEN_SIZE=14336
NUM_ATTN_HEADS=32
SEQ_LENGTH=4096

# --- MoE 相关参数 ---
NUM_EXPERTS=64
EXPERT_PARALLEL_SIZE=4 # 将64个专家分布在4个GPU上, 每个GPU 16个
MOE_ROUTED_TOKENS=4    # Top-k
MOE_CAPACITY_FACTOR=1.25 # 专家容量因子, 1.0表示刚好, >1.0提供冗余
MOE_LOSS_COEFF=0.01      # 负载均衡损失的权重
MOE_MIN_CAPACITY=4       # 每个专家的最小容量

# --- 训练超参数 ---
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=2048
TRAIN_TOKENS=10000000000000 # 10T tokens
LR=1.0e-4
MIN_LR=1.0e-5
LR_DECAY_STYLE="cosine"
LR_WARMUP_TOKENS=375000000 # 375B
WEIGHT_DECAY=0.1
GRAD_CLIP=1.0

# --- 精度配置 (TransformerEngine) ---
FP8_FORMAT="hybrid"
USE_FP8="--fp8-hybrid"

# --- 构建启动命令 ---
# 使用 torchrun 或 srun for Slurm
srun --jobid $SLURM_JOB_ID torchrun \
  --nnodes $NNODES \
  --nproc_per_node $GPUS_PER_NODE \
  --rdzv_id $SLURM_JOB_ID \
  --rdzv_backend c10d \
  --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
  pretrain_gpt.py \
  \
  # --- 分布式配置 ---
  --tensor-model-parallel-size $TP \
  --pipeline-model-parallel-size $PP \
  --expert-model-parallel-size $EXPERT_PARALLEL_SIZE \
  --distributed-backend nccl \
  # --- MoE 配置 ---
  --num-experts $NUM_EXPERTS \
  --moe-top-k $MOE_ROUTED_TOKENS \
  --moe-capacity-factor $MOE_CAPACITY_FACTOR \
  --moe-loss-coeff $MOE_LOSS_COEFF \
  --moe-min-capacity $MOE_MIN_CAPACITY \
  \
  # --- 模型架构 ---
  --num-layers $NUM_LAYERS \
  --hidden-size $HIDDEN_SIZE \
  --ffn-hidden-size $FFN_HIDDEN_SIZE \
  --num-attention-heads $NUM_ATTN_HEADS \
  --seq-length $SEQ_LENGTH \
  --max-position-embeddings $SEQ_LENGTH \
  --swiglu --use-rotary-position-embeddings --normalization RMSNorm \
  \
  # --- 训练参数 ---
  --micro-batch-size $MICRO_BATCH_SIZE \
  --global-batch-size $GLOBAL_BATCH_SIZE \
  --train-tokens $TRAIN_TOKENS \
  \
  # --- 精度与优化器 ---
  $USE_FP8 --bf16 \
  --optimizer adam --adam-beta1 0.9 --adam-beta2 0.95 --weight-decay $WEIGHT_DECAY \
  --grad-clip-norm $GRAD_CLIP \
  \
  # --- 学习率 ---
  --lr $LR --min-lr $MIN_LR --lr-decay-style $LR_DECAY_STYLE --lr-warmup-tokens $LR_WARMUP_TOKENS \
  \
  # --- 数据与IO ---
  --data-path $DATA_PATH \
  --tokenizer-type GPTSentencePieceTokenizer --tokenizer-model $TOKENIZER_PATH \
  --data-impl mmap --split 990,8,2 \
  \
  # --- 日志与保存 ---
  --save $CHECKPOINT_PATH --load $CHECKPOINT_PATH --save-interval 500 \
  --log-interval 10 --eval-interval 100 --eval-iters 10 \
  --wandb-project ${WANDB_PROJECT} \
  --num-workers 2 --seed 42
```

---

## A.4 监控仪表与告警规则

### A.4.1 Grafana 仪表盘设计 (关键面板)

一个好的仪表盘应提供全局概览和下钻细节的能力。

1.  **全局状态 (Overview)**
    *   **TFLOPs (Total & Per GPU):** 单值/仪表，显示当前集群总算力利用率。
    *   **Global Batch Loss:** 时间序列图，监控损失下降趋势。
    *   **Active Nodes / GPUs:** 状态面板，显示参与训练的节点数，异常节点标红。
    *   **Time to Completion (ETA):** 基于当前 tokens/sec 预估剩余训练时间。

2.  **性能诊断 (Performance Deep-Dive)**
    *   **GPU Utilization Heatmap:** X轴为时间，Y轴为 Rank ID，颜色深浅表示 GPU 利用率。快速定位掉队或空闲的 GPU。
    *   **FWD/BWD/OPT Time Breakdown:** 堆叠条形图，显示前向、反向和优化器更新的耗时占比。
    *   **Data Loader Time:** 时间序列图，监控数据加载是否成为瓶颈。
    *   **NVLink/IB Bandwidth:** 时间序列图，监控 AllReduce、AllGather 等通信操作的带宽。

3.  **MoE 专项监控 (MoE Dashboard)**
    *   **Expert Utilization Heatmap:** X轴为 Layer ID，Y轴为 Expert ID，颜色表示该专家被路由到的 token 百分比。用于发现“死亡专家”或负载热点。
    *   **Load Balancing Loss:** 时间序列图，监控 MoE 的辅助损失。
    *   **Router Z-Loss:** 时间序列图，监控路由 logits 的正则化损失。
    *   **Tokens Dropped per Layer:** 条形图，显示因容量不足而被丢弃的 token 数量。

### A.4.2 扩展的 Prometheus 告警规则 (`alerts.yml`)

```yaml
groups:
- name: megatron_training_alerts
  rules:
  # --- CRITICAL (需要立即响应) ---
  - alert: TrainingLossIsNaN
    # ... (同前)
  - alert: GpuEccDoubleBitError
    expr: dcgm_ecc_dbe_volatile_total{job="vla-10b-pretrain"} > 0
    for: 1m
    labels: { severity: "critical" }
    annotations:
      summary: "GPU出现双比特ECC错误 (instance: {{ $labels.instance }})"
      description: "节点 {{ $labels.instance }} 的 GPU {{ $labels.gpu }} 出现不可修复的内存错误，需隔离节点并检修。"

  # --- WARNING (需要关注，可能影响效率) ---
  - alert: ThroughputDroppedSignificantly
    # ... (同前，阈值可设为更感的 0.85)
  - alert: GradientNormExplosion
    expr: megatron_grad_norm{job="vla-10b-pretrain"} > 100.0
    for: 5m
    labels: { severity: "warning" }
    annotations:
      summary: "梯度范数爆炸 (job: {{ $labels.job }})"
      description: "模型 {{ $labels.job }} 的梯度范数持续高于 100.0，可能导致训练不稳定。"

  - alert: DeadMoeExpert
    expr: rate(megatron_moe_routed_tokens_per_expert{job="vla-10b-pretrain"}[15m]) == 0
    for: 30m
    labels: { severity: "warning" }
    annotations:
      summary: "检测到死亡的MoE专家 (layer: {{ $labels.layer }}, expert: {{ $labels.expert }})"
      description: "专家 ({{ $labels.layer }}, {{ $labels.expert }}) 在过去30分钟内没有被路由到任何 token。"

  # --- INFO (状态通知) ---
  - alert: LearningRateAnnealingStarted
    expr: megatron_lr{job="vla-10b-pretrain"} < 1.0e-4 # 假设这是初始LR
    for: 1m
    labels: { severity: "info" }
    annotations:
      summary: "学习率已开始退火 (job: {{ $labels.job }})"
      description: "训练已过 warmup 阶段，学习率调度器开始工作。"
```

---

## A.5 评测清单与打分汇总表模板

这张表更为详尽，加入了运行配置和预期耗时，使其成为一个可执行的评测计划。

| 维度 | 评测集 | 指标 | 1B Dense | 10B MoE | SOTA 参考 | 评测配置/命令 | 预期耗时 (A100x8) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **文本-理解** | MMLU | Acc | | | 89.7 (GPT-4) | `lm-eval --model ... --tasks mmlu --num_fewshot 5` | 4 hours |
| | C-Eval | Acc | | | 90.2 (GPT-4) | `lm-eval --tasks ceval_... --num_fewshot 5` | 3 hours |
| **文本-代码** | HumanEval | Pass@1 | | | 92.0 (AlphaCode2) | `lm-eval --tasks humaneval --batch_size 1` | 6 hours |
| **文本-长上下文** | Needle In A Haystack | Score | | | 99%+ | `run_niah.py --ctx_len 32768 --depths 10` | 2 hours |
| **音频-ASR** | AISHELL-1 (Zh) | CER | | | 4.1 (Whisper-L) | `asr_eval.py --dataset aishell1` | 1 hour |
| | LibriSpeech (En) | WER | | | 2.5 (Whisper-L) | `asr_eval.py --dataset librispeech --split test-clean` | 2 hours |
| **图像-理解** | MMBench | Acc | | | 78.3 (Qwen-VL-Max) | `mm_eval.py --dataset mmbench_test_en` | 1.5 hours |
| **视频-理解** | VATEX (En/Zh) | CIDEr | | | 80.1 (LLaMA-VID) | `video_eval.py --dataset vatex --task zh_captions` | 5 hours |
| **3D-理解** | Procedural QA | EM | | | - | `3d_eval.py --task blender_script_qa` | 0.5 hours |
| **VLA-具身** | CALVIN | Success Rate | | | 34% (RT-2) | `calvin_eval.py --checkpoint ...` | 24 hours |
| **VLA-自动驾驶** | nuScenes (Open-loop) | mAP/NDS | | | 75.8 (UniAD) | `nuscenes_eval.py --task detection` | 8 hours |
| **安全与对齐** | SafetyBench | Score | | | 95.2 (Claude 2) | `safety_eval.py --suite full` | 4 hours |
| | TruthfulQA | % Truthful | | | 60.0 (GPT-4) | `lm-eval --tasks truthfulqa_mc` | 1 hour |
