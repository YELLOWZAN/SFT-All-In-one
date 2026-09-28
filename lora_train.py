#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用 LoRA 微调脚本 —— 适用于任意 PyTorch (PT) 格式的 HuggingFace 模型。

功能特性：
  1. 自动识别数据集格式（src/tgt、messages、Alpaca、ShareGPT、raw text 等）
  2. 自动检测模型架构并推荐 LoRA 目标模块
  3. 自动合并 LoRA 权重并分片保存（可配置单文件大小上限）
  4. 所有超参数集中在 CONFIG 区域，便于调整

使用方式：
  1. 修改下方 CONFIG 区域的路径和超参数
  2. 运行: python lora_train.py
  3. 也可通过命令行覆盖: python lora_train.py --model_path /path/to/model --train_file /path/to/train.jsonl

=======================================================================
                   模型权重合并与分片详细说明
=======================================================================

一、为什么需要合并？

  LoRA 训练只保存了「增量权重」(adapter)，文件很小（通常 1-3GB），
  但它必须依赖原始基础模型才能运行。合并就是把 LoRA 增量权重「烘焙」
  进基础模型，得到一个可独立运行的完整模型。

  两种部署方式对比：
  ┌──────────────┬──────────────────────┬──────────────────────┐
  │              │ 方式A：合并后模型      │ 方式B：基础+Adapter    │
  ├──────────────┼──────────────────────┼──────────────────────┤
  │ 加载方式      │ 直接 from_pretrained  │ 基础 + PeftModel加载  │
  │ 文件大小      │ 完整模型大小(如40GB)   │ 基础模型 + adapter    │
  │ 推理速度      │ 最快（无额外计算）     │ 略慢（需计算LoRA增量）│
  │ 兼容性        │ 所有框架都支持        │ 需框架支持PEFT        │
  │ 灵活度        │ 固定一个版本          │ 可随时切换/叠加adapter│
  └──────────────┴──────────────────────┴──────────────────────┘

  本脚本默认采用方式A（合并），适合生产部署和vLLM等推理框架。

二、合并的核心原理

  LoRA 的数学表达为:  W' = W + (α/r) * B @ A
  其中 W 是原始权重矩阵，A、B 是 LoRA 训练得到的低秩矩阵，
  r 是 LoRA rank，α 是缩放系数。

  合并 (merge_and_unload) 执行的操作：
    1. 对每个 LoRA 目标层，计算 delta = (α/r) * B @ A
    2. 将 delta 加到原始权重 W 上：W_merged = W + delta
    3. 移除 LoRA 层（unload），模型变回普通的 nn.Linear
    4. 保存合并后的完整模型权重

  合并后模型的参数量与基础模型完全相同，只是权重数值发生了变化。

三、分片 (Sharding) 的作用

  大模型的权重文件可能非常大（如本任务的合并模型约40GB）。
  单文件存储和传输存在以下问题：
    - 部分文件系统/传输协议对单文件大小有限制
    - 下载失败时需要重传整个文件
    - 并行加载多文件比单文件更快

  分片将一个大权重文件拆分为多个小文件，通过 index.json 索引。
  HuggingFace 的 save_pretrained 支持 max_shard_size 参数自动分片。

  分片后的目录结构示例：
    final_output/
    ├── config.json                    # 模型配置
    ├── tokenizer.json / *.model       # 分词器文件
    ├── model.safetensors.index.json   # 分片索引（记录每个权重在哪个文件）
    ├── model-00001-of-00010.safetensors  # 分片1
    ├── model-00002-of-00010.safetensors  # 分片2
    ├── ...
    └── model-00010-of-00010.safetensors  # 分片10

  推理框架（如vLLM）会自动读取 index.json 并加载所有分片，
  对用户完全透明，加载方式与单文件模型完全相同。

四、shard_max_size 参数

  控制每个分片文件的最大大小，支持以下格式：
    "5GB"    -> 5 * 1024^3 bytes
    "5000MB" -> 5000 * 1024^2 bytes
    "1024"   -> 1024 bytes (纯数字视为字节)

  常见取值建议：
    - "5GB"  : 通用推荐，适合大多数场景和传输限制
    - "10GB" : 减少文件数量，适合本地存储
    - None   : 不分片，保存为单个 model.safetensors

  注意：实际分片大小可能略小于设定值，因为权重是按层分配的。

五、合并时的显存管理

  合并过程需要同时加载：
    - 基础模型权重（如21B模型约40GB bf16）
    - LoRA adapter（约1-3GB）

  本脚本在合并前会释放训练时的模型显存：
    del model
    torch.cuda.empty_cache()

  然后重新加载基础模型进行合并。如果 GPU 显存不足，
  可将 device_map 改为 "cpu"（需足够CPU内存）或使用
  --no_merge 跳过合并，之后用独立脚本在显存充足时合并。

六、合并后验证

  合并完成后，建议用 vLLM 或 transformers 进行推理验证：
    from vllm import LLM
    llm = LLM(model=cfg["final_output"], trust_remote_code=True)
    outputs = llm.generate(["你好"])
  确保模型能正常加载并生成合理输出。

"""

import os
import sys
import json
import math
import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, PeftModel, TaskType


# =====================================================================
#                              CONFIG 区域
#           修改这里的参数即可适配不同模型和数据集
# =====================================================================

CONFIG = {
    # ---- 路径 ----
    "model_path": "/root/autodl-tmp/ERNIEPT",          # 基础模型路径 (PT格式)
    "train_file": "/root/autodl-tmp/sft_train.jsonl",  # 训练集 (jsonl)
    "val_file": "/root/autodl-tmp/sft_val.jsonl",      # 验证集 (jsonl)
    "output_dir": "/root/autodl-tmp/train_file",       # LoRA checkpoint 输出
    "final_output": "/root/autodl-tmp/train_output",    # 合并后完整模型输出

    # ---- LoRA 参数 ----
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.0,        # 部分模型(如MoE)需设为0
    "lora_target_modules": None,  # None=自动检测; 也可手动指定如 ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
    "lora_bias": "none",

    # ---- 训练超参数 ----
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

    # ---- 输出控制 ----
    # merge_after_train:
    #   True  = 训练结束后自动将 LoRA 增量权重合并到基础模型，输出可直接部署的完整模型
    #   False = 仅保存 LoRA adapter，后续可手动合并（见脚本 --no_merge 提示）
    "merge_after_train": True,
    # shard_max_size: 合并模型的分片大小上限
    #   取值示例: "5GB" / "10GB" / "5000MB" / None
    #   None 表示不分片，保存为单个 model.safetensors 文件
    #   详细说明见脚本头部「模型权重合并与分片详细说明」
    "shard_max_size": "5GB",
    # dtype: 训练和合并时的数据精度
    #   bfloat16: 推荐，A100/A800/H100等显卡原生支持
    #   float16:  老显卡(如V100)使用，可能需要梯度缩放
    #   float32:  全精度，显存占用翻倍，一般不用于大模型训练
    "dtype": "bfloat16",
}

# =====================================================================
#                         数据集格式自动识别
# =====================================================================

def detect_dataset_format(example):
    """
    自动识别一条样本的数据格式，返回格式名称。

    支持的格式：
      - src_tgt:      {"src": str|list, "tgt": str|list}
      - messages:     {"messages": [{"role":..., "content":...}, ...]}
      - alpaca:       {"instruction": str, "input": str(可选), "output": str}
      - prompt_response: {"prompt": str, "response": str} 或 {"question":..., "answer":...}
      - sharegpt:     {"conversations": [{"from": "human"|"gpt", "value": str}, ...]}
      - raw_text:     {"text": str} 或 {"content": str}
    """
    keys = set(example.keys())

    if "src" in keys and "tgt" in keys:
        return "src_tgt"
    if "messages" in keys:
        return "messages"
    if "conversations" in keys:
        return "sharegpt"
    if "instruction" in keys and "output" in keys:
        return "alpaca"
    if "prompt" in keys and "response" in keys:
        return "prompt_response"
    if "question" in keys and "answer" in keys:
        return "prompt_response"
    if "text" in keys or "content" in keys:
        return "raw_text"
    return "unknown"


def extract_messages(example, fmt):
    """
    将各种格式的样本统一转换为 OpenAI messages 格式:
    [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    """
    def _to_str(v):
        if isinstance(v, list):
            return v[0] if len(v) > 0 else ""
        return str(v) if v is not None else ""

    if fmt == "src_tgt":
        return [
            {"role": "user", "content": _to_str(example["src"])},
            {"role": "assistant", "content": _to_str(example["tgt"])},
        ]

    if fmt == "messages":
        return example["messages"]

    if fmt == "sharegpt":
        role_map = {"human": "user", "gpt": "assistant", "system": "system"}
        msgs = []
        for turn in example["conversations"]:
            role = role_map.get(turn.get("from", ""), turn.get("from", "user"))
            msgs.append({"role": role, "content": turn.get("value", "")})
        return msgs

    if fmt == "alpaca":
        instr = _to_str(example["instruction"])
        inp = _to_str(example.get("input", ""))
        prompt = instr + (f"\n{inp}" if inp else "")
        return [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": _to_str(example["output"])},
        ]

    if fmt == "prompt_response":
        q = example.get("prompt", example.get("question", ""))
        a = example.get("response", example.get("answer", ""))
        return [
            {"role": "user", "content": _to_str(q)},
            {"role": "assistant", "content": _to_str(a)},
        ]

    if fmt == "raw_text":
        text = _to_str(example.get("text", example.get("content", "")))
        return [{"role": "user", "content": text}]

    # 未知格式：尝试用所有值拼接
    raise ValueError(f"未知数据格式，样本字段: {list(example.keys())}")


def get_prompt_messages(messages):
    """从完整 messages 中提取 prompt 部分（最后一个 assistant 之前的内容）。"""
    prompt_msgs = []
    for m in messages:
        if m["role"] == "assistant":
            break
        prompt_msgs.append(m)
    return prompt_msgs


# =====================================================================
#                         LoRA 目标模块自动检测
# =====================================================================

# 常见注意力投影和MLP层名称模式
COMMON_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj", "gate_up_proj",
    "query_key_value", "qkv_proj", "dense",
    "c_fc", "c_proj", "fc_in", "fc_out",
    "W_pack", "o_proj", "mlp",
]


def detect_target_modules(model):
    """
    扫描模型模块，自动识别适合做 LoRA 的线性层名称。
    优先匹配常见的注意力投影和 MLP 层。
    """
    found = set()
    for name, module in model.named_modules():
        # 只关注 Linear 层
        if not isinstance(module, torch.nn.Linear):
            continue
        # 取模块名的最后一段作为目标
        short_name = name.split(".")[-1]
        if short_name in COMMON_TARGET_MODULES:
            found.add(short_name)

    if found:
        return sorted(found)

    # 兜底：返回所有线性层的末级名称（去重）
    all_linear = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            all_linear.add(name.split(".")[-1])
    return sorted(all_linear)


# =====================================================================
#                         数据处理函数
# =====================================================================

def make_tokenize_fn(tokenizer, max_len):
    """构建分词函数，自动识别格式并做 label masking。"""

    def tokenize_fn(batch):
        input_ids_list = []
        labels_list = []
        attention_mask_list = []

        # 检测格式（取第一条样本）
        first = {k: batch[k][0] for k in batch.keys()}
        fmt = detect_dataset_format(first)

        n = len(batch[list(batch.keys())[0]])
        for i in range(n):
            example = {k: batch[k][i] for k in batch.keys()}
            try:
                messages = extract_messages(example, fmt)
            except ValueError:
                continue

            # 完整对话文本
            full_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            # prompt 文本（不含 assistant 回复）
            prompt_msgs = get_prompt_messages(messages)
            prompt_text = tokenizer.apply_chat_template(
                prompt_msgs, tokenize=False, add_generation_prompt=True
            )

            full_ids = tokenizer(full_text, add_special_tokens=False).input_ids
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids

            # 截断
            if len(full_ids) > max_len:
                full_ids = full_ids[:max_len]

            # label masking: prompt 部分用 -100
            labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
            if len(labels) < len(full_ids):
                labels = labels + full_ids[len(labels):]
            labels = labels[:len(full_ids)]

            # 确保末尾有 eos
            if full_ids[-1] != tokenizer.eos_token_id:
                full_ids.append(tokenizer.eos_token_id)
                labels.append(tokenizer.eos_token_id)

            input_ids_list.append(full_ids)
            labels_list.append(labels)
            attention_mask_list.append([1] * len(full_ids))

        return {
            "input_ids": input_ids_list,
            "labels": labels_list,
            "attention_mask": attention_mask_list,
        }

    return tokenize_fn


# =====================================================================
#                              主流程
# =====================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="通用 LoRA 微调脚本")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--train_file", type=str, default=None)
    parser.add_argument("--val_file", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--final_output", type=str, default=None)
    parser.add_argument("--max_seq_length", type=int, default=None)
    parser.add_argument("--lora_r", type=int, default=None)
    parser.add_argument("--lora_alpha", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--grad_accum", type=int, default=None)
    parser.add_argument("--shard_max_size", type=str, default=None)
    parser.add_argument("--no_merge", action="store_true", help="训练后不合并权重")
    return parser.parse_args()


def main():
    args = parse_args()

    # 合并 CONFIG 和命令行参数
    cfg = dict(CONFIG)
    arg_map = {
        "model_path": args.model_path,
        "train_file": args.train_file,
        "val_file": args.val_file,
        "output_dir": args.output_dir,
        "final_output": args.final_output,
        "max_seq_length": args.max_seq_length,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "learning_rate": args.learning_rate,
        "num_epochs": args.num_epochs,
        "per_device_batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum,
        "shard_max_size": args.shard_max_size,
    }
    for k, v in arg_map.items():
        if v is not None:
            cfg[k] = v
    if args.no_merge:
        cfg["merge_after_train"] = False

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map.get(cfg["dtype"], torch.bfloat16)

    print("=" * 60)
    print(f"通用 LoRA 微调 | 模型: {cfg['model_path']}")
    print("=" * 60)

    # ---- 1. 加载 tokenizer ----
    print("\n[1/6] 加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_path"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = "right"
    print(f"  vocab_size={tokenizer.vocab_size}, pad={repr(tokenizer.pad_token)}, eos={repr(tokenizer.eos_token)}")

    # ---- 2. 加载模型 ----
    print(f"\n[2/6] 加载模型 ({cfg['dtype']})...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_path"],
        dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  总参数量: {total_params/1e9:.2f}B")
    print(f"  模型架构: {model.config.architectures}")

    # ---- 3. 确定 LoRA 目标模块 ----
    print("\n[3/6] 配置 LoRA...")
    target_modules = cfg["lora_target_modules"]
    if target_modules is None:
        target_modules = detect_target_modules(model)
        print(f"  自动检测目标模块: {target_modules}")
    else:
        print(f"  使用指定目标模块: {target_modules}")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=target_modules,
        bias=cfg["lora_bias"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ---- 4. 加载并处理数据集 ----
    print("\n[4/6] 加载和处理数据集...")
    dataset = load_dataset(
        "json",
        data_files={"train": cfg["train_file"], "validation": cfg["val_file"]},
    )
    print(f"  原始 -> 训练集: {len(dataset['train'])}, 验证集: {len(dataset['validation'])}")

    # 检测数据格式
    sample = dataset["train"][0]
    fmt = detect_dataset_format(sample)
    print(f"  检测到数据格式: {fmt}")

    tokenize_fn = make_tokenize_fn(tokenizer, cfg["max_seq_length"])
    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        batch_size=32,
        remove_columns=dataset["train"].column_names,
        num_proc=4,
    )
    tokenized = tokenized.filter(lambda x: len(x["input_ids"]) > 10)
    print(f"  过滤后 -> 训练集: {len(tokenized['train'])}, 验证集: {len(tokenized['validation'])}")

    # ---- 5. 训练 ----
    print("\n[5/6] 开始训练...")
    steps_per_epoch = math.ceil(
        len(tokenized["train"]) / (cfg["per_device_batch_size"] * cfg["grad_accum_steps"])
    )
    total_steps = steps_per_epoch * cfg["num_epochs"]
    warmup_steps = int(cfg["warmup_ratio"] * total_steps)
    print(f"  steps/epoch={steps_per_epoch}, total_steps={total_steps}, warmup_steps={warmup_steps}")

    training_args = TrainingArguments(
        output_dir=cfg["output_dir"],
        num_train_epochs=cfg["num_epochs"],
        per_device_train_batch_size=cfg["per_device_batch_size"],
        per_device_eval_batch_size=cfg["per_device_batch_size"],
        gradient_accumulation_steps=cfg["grad_accum_steps"],
        learning_rate=cfg["learning_rate"],
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        logging_steps=cfg["logging_steps"],
        save_steps=cfg["save_steps"],
        save_total_limit=cfg["save_total_limit"],
        eval_steps=cfg["eval_steps"],
        eval_strategy="steps",
        bf16=(cfg["dtype"] == "bfloat16"),
        fp16=(cfg["dtype"] == "float16"),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",
        report_to="none",
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=data_collator,
    )

    trainer.train()

    # 保存 LoRA adapter
    print("\n训练完成，保存 LoRA adapter...")
    model.save_pretrained(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])
    print(f"  LoRA adapter 已保存到 {cfg['output_dir']}")

    # ---- 6. 合并 LoRA 权重 ----
    if cfg["merge_after_train"]:
        print("\n[6/6] 合并 LoRA 权重到基础模型...")

        # 释放训练模型占用的显存，为重新加载基础模型腾出空间
        del model
        torch.cuda.empty_cache()

        # 重新加载基础模型（干净权重，不含LoRA）
        # 注意：必须重新加载，因为训练时的 model 已被 PEFT 包装
        base_model = AutoModelForCausalLM.from_pretrained(
            cfg["model_path"],
            dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        )

        # 将训练好的 LoRA adapter 加载到基础模型上
        # 此时 base_model 的每个目标 Linear 层被替换为 LoraLayer
        merged_model = PeftModel.from_pretrained(base_model, cfg["output_dir"])

        # 执行合并：
        #   1. 对每个 LoRA 层计算 delta = (lora_alpha / r) * B @ A
        #   2. 将 delta 加到原始权重：W_merged = W + delta
        #   3. unload: 移除 LoRA 包装层，恢复为普通 nn.Linear
        # 合并后模型结构与基础模型完全一致，可被任何推理框架直接加载
        merged_model = merged_model.merge_and_unload()

        # 准备保存参数
        save_kwargs = {"safe_serialization": True}
        if cfg["shard_max_size"]:
            # max_shard_size: 单个权重文件的最大字节数
            # 超过此大小的模型会被自动拆分为多个 .safetensors 分片文件
            # 同时生成 model.safetensors.index.json 索引文件
            save_kwargs["max_shard_size"] = cfg["shard_max_size"]

        os.makedirs(cfg["final_output"], exist_ok=True)

        # 保存合并后的模型权重（自动分片）
        merged_model.save_pretrained(cfg["final_output"], **save_kwargs)

        # 保存分词器文件（config.json, tokenizer.json, vocab 等）
        tokenizer.save_pretrained(cfg["final_output"])

        print(f"  合并模型已保存到 {cfg['final_output']}")

        # 如果启用了分片，打印各分片文件的大小统计
        if cfg["shard_max_size"]:
            print(f"  分片大小上限: {cfg['shard_max_size']}")
            total = 0
            for f in sorted(os.listdir(cfg["final_output"])):
                if f.endswith(".safetensors"):
                    size_mb = os.path.getsize(os.path.join(cfg["final_output"], f)) / (1024 * 1024)
                    total += size_mb
                    print(f"    {f:45s} {size_mb:8.1f} MB")
            print(f"    {'TOTAL':45s} {total:8.1f} MB ({total/1024:.1f} GB)")
            print(f"\n  提示: 推理框架会自动读取 model.safetensors.index.json 加载全部分片")
    else:
        print("\n[6/6] 跳过合并 (--no_merge)")
        print(f"  LoRA adapter 保留在 {cfg['output_dir']}")
        print(f"  如需合并，可后续运行:")
        print(f"    python -c \"from peft import PeftModel; from transformers import AutoModelForCausalLM; "
              f"m=AutoModelForCausalLM.from_pretrained('{cfg['model_path']}', dtype='auto', device_map='auto'); "
              f"m=PeftModel.from_pretrained(m, '{cfg['output_dir']}').merge_and_unload(); "
              f"m.save_pretrained('{cfg['final_output']}', max_shard_size='5GB')\"")

    print("\n" + "=" * 60)
    print("全部完成!")
    print(f"  LoRA checkpoints: {cfg['output_dir']}")
    if cfg["merge_after_train"]:
        print(f"  合并后模型: {cfg['final_output']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
