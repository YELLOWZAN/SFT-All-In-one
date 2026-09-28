# 通用 LoRA 微调脚本使用说明

> 脚本位置：`任务脚本/lora_train.py`

本文档详细说明通用 LoRA 微调脚本的功能、配置、使用方法和注意事项。该脚本可复用于任意 **PyTorch (PT) 格式**的 HuggingFace 大语言模型，支持自动识别数据集格式、自动检测 LoRA 目标模块、自动合并权重并分片保存。

---

## 目录

1. [功能概述](#一功能概述)
2. [环境依赖](#二环境依赖)
3. [快速开始](#三快速开始)
4. [配置参数详解](#四配置参数详解)
5. [命令行参数](#五命令行参数)
6. [数据集格式支持](#六数据集格式支持)
7. [LoRA 目标模块自动检测](#七lora-目标模块自动检测)
8. [训练流程说明](#八训练流程说明)
9. [模型权重合并与分片](#九模型权重合并与分片)
10. [输出产物](#十输出产物)
11. [常见问题](#十一常见问题)
12. [高级用法](#十二高级用法)

---

## 一、功能概述

本脚本实现了一个端到端的 LoRA（Low-Rank Adaptation）监督微调流程，核心能力：

| 能力 | 说明 |
|------|------|
| **通用模型兼容** | 支持任意 PT 格式 HuggingFace 模型（Llama、Qwen、ERNIE、ChatGLM 等） |
| **数据集格式自动识别** | 自动检测并支持 6 种常见指令微调数据格式 |
| **LoRA 目标模块自动检测** | 扫描模型 Linear 层，自动匹配注意力投影和 MLP 层 |
| **自动合并权重** | 训练后自动将 LoRA 增量合并到基础模型，输出可直接部署的完整模型 |
| **自动分片保存** | 按设定大小上限将大模型权重拆分为多个文件，便于传输 |
| **灵活配置** | 所有超参数集中在 CONFIG 区域，也支持命令行覆盖 |

---

## 二、环境依赖

### Python 包

```bash
pip install torch transformers peft datasets accelerate
```

| 包 | 最低版本 | 用途 |
|----|----------|------|
| torch | 2.0+ | 深度学习框架 |
| transformers | 4.40+ | 模型加载与训练 (5.x 也支持) |
| peft | 0.10+ | LoRA 实现 |
| datasets | 2.14+ | 数据集加载 |
| accelerate | 0.27+ | 训练加速与混合精度 |

### 硬件要求

- **GPU**: 支持 CUDA 的 NVIDIA GPU（建议显存 ≥ 24GB，大模型需 40GB+）
- **磁盘**: 根据模型大小，建议预留 2~3 倍模型大小的空间
- **内存**: 合并模型时建议 ≥ 模型大小 + 10GB

### 数据类型选择

| dtype | 适用 GPU | 说明 |
|-------|----------|------|
| bfloat16 | A100/A800/H100/RTX 30/40 系 | 推荐，数值稳定，无需梯度缩放 |
| float16 | V100/T4 | 老显卡使用，可能需要梯度缩放 |
| float32 | 任意 | 全精度，显存占用翻倍，一般不用于大模型 |

---

## 三、快速开始

### 步骤 1：修改配置

打开 `任务脚本/lora_train.py`，修改 `CONFIG` 区域：

```python
CONFIG = {
    "model_path": "/path/to/your/base_model",   # 基础模型路径
    "train_file": "/path/to/train.jsonl",         # 训练集
    "val_file": "/path/to/val.jsonl",             # 验证集
    "output_dir": "/path/to/lora_output",         # LoRA checkpoint 输出
    "final_output": "/path/to/merged_model",      # 合并后完整模型输出
    # ... 其他参数按需调整
}
```

### 步骤 2：运行训练

```bash
cd 任务脚本
python lora_train.py
```

### 步骤 3：查看产物

训练完成后，脚本会自动：
1. 保存 LoRA adapter 到 `output_dir`
2. 合并权重并分片保存到 `final_output`

---

## 四、配置参数详解

所有参数定义在脚本顶部的 `CONFIG` 字典中。

### 路径配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model_path` | str | `/root/autodl-tmp/ERNIEPT` | 基础模型路径（PT 格式，需含 config.json 和权重文件） |
| `train_file` | str | `/root/autodl-tmp/sft_train.jsonl` | 训练集 JSONL 文件路径 |
| `val_file` | str | `/root/autodl-tmp/sft_val.jsonl` | 验证集 JSONL 文件路径 |
| `output_dir` | str | `/root/autodl-tmp/train_file` | LoRA adapter 和训练 checkpoint 输出目录 |
| `final_output` | str | `/root/autodl-tmp/train_output` | 合并后完整模型输出目录 |

### LoRA 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `lora_r` | int | 16 | LoRA 秩(rank)，越大拟合能力越强但参数越多，常用 8/16/32/64 |
| `lora_alpha` | int | 32 | LoRA 缩放系数，通常设为 r 的 2 倍 |
| `lora_dropout` | float | 0.0 | LoRA 层 dropout，MoE 等特殊架构需设为 0.0 |
| `lora_target_modules` | list/None | None | LoRA 目标层名称列表，None 表示自动检测 |
| `lora_bias` | str | "none" | 是否训练 bias，可选 "none"/"all"/"lora_only" |

### 训练超参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `max_seq_length` | int | 2048 | 最大序列长度，超过会被截断 |
| `learning_rate` | float | 2e-4 | 学习率，LoRA 常用 1e-4 ~ 3e-4 |
| `num_epochs` | int | 1 | 训练轮数 |
| `per_device_batch_size` | int | 4 | 单卡 batch size |
| `grad_accum_steps` | int | 4 | 梯度累积步数，有效 batch = batch_size × grad_accum |
| `warmup_ratio` | float | 0.03 | warmup 步数占总步数的比例 |
| `save_steps` | int | 500 | 每隔多少步保存一次 checkpoint |
| `eval_steps` | int | 500 | 每隔多少步评估一次 |
| `logging_steps` | int | 10 | 每隔多少步打印训练日志 |
| `save_total_limit` | int | 3 | 最多保留多少个 checkpoint |

### 输出控制

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `merge_after_train` | bool | True | 训练后是否自动合并 LoRA 权重 |
| `shard_max_size` | str/None | "5GB" | 合并模型分片大小上限，None 表示不分片 |
| `dtype` | str | "bfloat16" | 训练和合并的数据精度 |

---

## 五、命令行参数

除了修改 CONFIG，也可通过命令行参数覆盖配置（优先级高于 CONFIG）：

```bash
python lora_train.py \
  --model_path /path/to/model \
  --train_file /path/to/train.jsonl \
  --val_file /path/to/val.jsonl \
  --output_dir /path/to/lora_out \
  --final_output /path/to/merged \
  --max_seq_length 2048 \
  --lora_r 16 \
  --lora_alpha 32 \
  --learning_rate 2e-4 \
  --num_epochs 1 \
  --batch_size 4 \
  --grad_accum 4 \
  --shard_max_size 5GB \
  --no_merge
```

| 参数 | 说明 |
|------|------|
| `--model_path` | 基础模型路径 |
| `--train_file` | 训练集路径 |
| `--val_file` | 验证集路径 |
| `--output_dir` | LoRA 输出目录 |
| `--final_output` | 合并模型输出目录 |
| `--max_seq_length` | 最大序列长度 |
| `--lora_r` | LoRA rank |
| `--lora_alpha` | LoRA alpha |
| `--learning_rate` | 学习率 |
| `--num_epochs` | 训练轮数 |
| `--batch_size` | 单卡 batch size |
| `--grad_accum` | 梯度累积步数 |
| `--shard_max_size` | 分片大小上限 |
| `--no_merge` | 训练后不合并权重（仅保存 adapter） |

---

## 六、数据集格式支持

脚本会自动识别数据格式，无需手动指定。所有样本最终统一转换为 OpenAI `messages` 格式处理。

### 支持的格式

#### 1. src/tgt 格式

```json
{"src": "你好", "tgt": "你好！有什么可以帮你的吗？"}
{"src": ["用户指令"], "tgt": ["助手回复"]}
```

#### 2. OpenAI messages 格式

```json
{
  "messages": [
    {"role": "system", "content": "你是一个助手"},
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！"}
  ]
}
```

#### 3. Alpaca 格式

```json
{"instruction": "翻译成英文", "input": "你好", "output": "Hello"}
{"instruction": "解释什么是机器学习", "input": "", "output": "机器学习是..."}
```

#### 4. Prompt/Response 格式

```json
{"prompt": "你好", "response": "你好！"}
{"question": "1+1等于几", "answer": "2"}
```

#### 5. ShareGPT 格式

```json
{
  "conversations": [
    {"from": "human", "value": "你好"},
    {"from": "gpt", "value": "你好！"}
  ]
}
```

#### 6. Raw text 格式

```json
{"text": "用户：你好\n助手：你好！"}
{"content": "用户：你好\n助手：你好！"}
```

> **注意**：raw_text 格式会将整段文本作为 user 输入，不会做 label masking，适合预训练继续训练场景。

### 数据集文件要求

- 格式：JSONL（每行一个 JSON 对象）
- 编码：UTF-8
- 训练集和验证集必须使用相同格式

### Label Masking 策略

脚本会自动对 assistant 回复之外的 token 进行 mask（label 设为 -100），只计算 assistant 回复部分的 loss：

```
完整文本:  [system] [user prompt] [assistant response] <eos>
label:      -100     -100              正常计算 loss      <eos>
```

---

## 七、LoRA 目标模块自动检测

当 `lora_target_modules` 设为 `None` 时，脚本会自动扫描模型中的 `nn.Linear` 层，匹配以下常见层名：

```
q_proj, k_proj, v_proj, o_proj,           # 注意力投影
gate_proj, up_proj, down_proj,            # MLP 层 (Llama/Qwen 风格)
gate_up_proj, query_key_value, qkv_proj,  # 融合层
c_fc, c_proj,                              # GPT-2 风格
fc_in, fc_out, dense,                      # 其他常见命名
W_pack, mlp                                # ChatGLM 等
```

### 手动指定目标模块

如需手动指定（例如只想微调注意力层），在 CONFIG 中设置：

```python
"lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
```

### 自动检测输出示例

```
[3/6] 配置 LoRA...
  自动检测目标模块: ['down_proj', 'gate_proj', 'k_proj', 'o_proj', 'q_proj', 'up_proj', 'v_proj']
  trainable params: 52,428,800 || all params: 21,831,204,864 || trainable%: 0.2402
```

---

## 八、训练流程说明

脚本执行 6 个步骤：

```
[1/6] 加载 tokenizer
      → 自动设置 pad_token，padding_side="right"

[2/6] 加载模型
      → dtype=bfloat16, device_map="auto"
      → 开启 gradient_checkpointing 节省显存
      → enable_input_require_grads (PEFT 需要)

[3/6] 配置 LoRA
      → 自动检测或使用指定的 target_modules
      → 用 get_peft_model 包装模型

[4/6] 加载和处理数据集
      → 自动识别数据格式
      → apply_chat_template 生成对话文本
      → tokenize + label masking
      → 过滤过短样本 (< 10 tokens)

[5/6] 训练
      → Trainer + DataCollatorForSeq2Seq
      → cosine 学习率调度
      → adamw_torch_fused 优化器
      → 每 save_steps 保存 checkpoint
      → 每 eval_steps 在验证集上评估

[6/6] 合并 LoRA 权重 (若 merge_after_train=True)
      → 释放训练显存
      → 重新加载基础模型
      → PeftModel.from_pretrained + merge_and_unload
      → save_pretrained (自动分片)
```

### 显存优化策略

- `gradient_checkpointing`: 用计算换显存，减少激活值存储
- `dtype=bfloat16`: 权重和激活用半精度
- `per_device_batch_size=4` + `grad_accum=4`: 等效 batch_size=16
- 训练后 `del model` + `torch.cuda.empty_cache()` 释放显存供合并使用

---

## 九、模型权重合并与分片

### 为什么需要合并

LoRA 训练只保存增量权重（adapter，通常 1~3GB），必须依赖基础模型才能运行。合并就是将 LoRA 增量「烘焙」进基础模型，得到可独立运行的完整模型。

### 合并的数学原理

```
W' = W + (α/r) × B @ A

W:  原始权重矩阵
A:  LoRA 降维矩阵 (r × d)
B:  LoRA 升维矩阵 (d × r)
r:  LoRA rank
α:  缩放系数
```

`merge_and_unload()` 执行：
1. 计算 `delta = (α/r) × B @ A`
2. `W_merged = W + delta`
3. 移除 LoRA 包装层，恢复为普通 `nn.Linear`

### 分片机制

大模型权重（如 40GB）通过 `max_shard_size` 参数拆分为多个文件：

```
final_output/
├── config.json
├── tokenizer.json / tokenizer.model
├── model.safetensors.index.json      # 分片索引
├── model-00001-of-00010.safetensors  # 分片 1
├── model-00002-of-00010.safetensors  # 分片 2
├── ...
└── model-00010-of-00010.safetensors  # 分片 10
```

推理框架（vLLM、transformers）会自动读取 `index.json` 加载全部分片，对用户完全透明。

### shard_max_size 取值

| 值 | 说明 |
|----|------|
| `"5GB"` | 推荐，适合大多数传输场景 |
| `"10GB"` | 减少文件数量，适合本地存储 |
| `None` | 不分片，保存为单个 `model.safetensors` |

---

## 十、输出产物

### output_dir（LoRA checkpoint）

```
output_dir/
├── adapter_config.json          # LoRA 配置 (r, alpha, target_modules 等)
├── adapter_model.safetensors    # LoRA 增量权重
├── tokenizer.json / .model      # 分词器文件
├── special_tokens_map.json
├── trainer_state.json           # 训练状态 (loss, lr, steps)
├── checkpoint-XXX/              # 训练 checkpoint
│   ├── adapter_model.safetensors
│   ├── optimizer.pt
│   └── ...
└── train.log                    # 训练日志
```

### final_output（合并后完整模型）

```
final_output/
├── config.json                  # 模型配置
├── model.safetensors.index.json # 分片索引
├── model-00001-of-XXXXX.safetensors
├── ...
├── tokenizer.json / .model
├── special_tokens_map.json
└── tokenizer_config.json
```

### 部署方式对比

| 方式 | 加载命令 | 优点 | 缺点 |
|------|----------|------|------|
| 合并模型 | `AutoModelForCausalLM.from_pretrained(final_output)` | 所有框架支持，推理最快 | 文件大，固定一个版本 |
| Adapter | `PeftModel.from_pretrained(base_model, output_dir)` | 文件小，可切换/叠加 | 需框架支持 PEFT，推理略慢 |

---

## 十一、常见问题

### Q1: 显存不足 (OOM) 怎么办？

**解决方案**（按优先级）：
1. 减小 `per_device_batch_size`（如 4→2）
2. 增大 `grad_accum_steps` 保持有效 batch size
3. 减小 `max_seq_length`
4. 确保 `gradient_checkpointing=True`（默认已开启）
5. 合并时 OOM：确保训练结束后释放了显存，或用 `--no_merge` 后单独合并

### Q2: LoRA dropout 报错

部分模型（如 MoE 架构的 ERNIE）使用了 `ParamWrapper`，不支持非零 dropout。

**解决**: 设置 `"lora_dropout": 0.0`

### Q3: warmup_ratio 报错（transformers 5.x）

transformers 5.x 移除了 `warmup_ratio`，改用 `warmup_steps`。

**本脚本已自动处理**：`warmup_steps = int(warmup_ratio * total_steps)`

### Q4: 数据格式识别失败

如果数据格式不在支持列表中，脚本会抛出 `ValueError: 未知数据格式`。

**解决**: 
- 检查数据字段名是否匹配支持的格式
- 或将数据转换为 `messages` 格式（最通用）

### Q5: 如何只保存 adapter 不合并？

运行时加 `--no_merge` 参数，或在 CONFIG 中设置 `"merge_after_train": False`。

后续手动合并：
```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("base_model_path", dtype="auto", device_map="auto")
model = PeftModel.from_pretrained(base, "lora_output_dir").merge_and_unload()
model.save_pretrained("merged_output", max_shard_size="5GB")
```

### Q6: 如何恢复训练？

从最后一个 checkpoint 恢复：
```python
trainer.train(resume_from_checkpoint="/path/to/output_dir/checkpoint-XXXX")
```
（需自行修改脚本或使用 Trainer API）

---

## 十二、高级用法

### 1. 自定义 LoRA 目标模块

```python
CONFIG = {
    # 只微调注意力投影层
    "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
}
```

### 2. 多轮对话数据

`messages` 和 `sharegpt` 格式天然支持多轮对话，label masking 会自动对所有 assistant 轮次计算 loss。

### 3. 大模型分片加载

基础模型本身可以是分片的（多个 `.safetensors` + `index.json`），`from_pretrained` 会自动加载。

### 4. 使用本地 tokenizer

如果模型目录已包含 tokenizer 文件，脚本会自动加载。如需单独指定，可修改代码中的 tokenizer 路径。

### 5. 调整保存频率

对于大模型训练，可增大 `save_steps` 和 `eval_steps` 以减少保存开销：
```python
"save_steps": 1000,
"eval_steps": 1000,
```

---

## 附录：完整配置示例

```python
CONFIG = {
    # 路径
    "model_path": "/root/autodl-tmp/ERNIEPT",
    "train_file": "/root/autodl-tmp/sft_train.jsonl",
    "val_file": "/root/autodl-tmp/sft_val.jsonl",
    "output_dir": "/root/autodl-tmp/train_file",
    "final_output": "/root/autodl-tmp/train_output",

    # LoRA
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.0,
    "lora_target_modules": None,  # 自动检测
    "lora_bias": "none",

    # 训练
    "max_seq_length": 2048,
    "learning_rate": 2e-4,
    "num_epochs": 1,
    "per_device_batch_size": 4,
    "grad_accum_steps": 4,
    "warmup_ratio": 0.03,
    "save_steps": 500,
    "eval_steps": 500,
    "logging_steps": 10,
    "save_total_limit": 3,

    # 输出
    "merge_after_train": True,
    "shard_max_size": "5GB",
    "dtype": "bfloat16",
}
```
