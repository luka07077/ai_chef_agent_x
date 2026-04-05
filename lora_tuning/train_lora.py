"""
御厨·臻享 QLoRA 微调训练脚本
========================================
用途：对 Qwen2.5-7B-Instruct 进行 QLoRA（4-bit NF4）微调，使其更贴合"御厨·臻享"私厨人设。
注意：需要 CUDA GPU（Colab T4 16GB 即可），不支持 MPS/CPU。

运行方式：
    cd /Users/luka/Code/python/ai_chef_agent
    python lora_tuning/train_lora.py

前置依赖（建议在 venv 中安装）：
    pip install transformers>=4.43.0 peft>=0.12.0 trl>=0.10.0 datasets accelerate bitsandbytes tqdm

训练数据格式（data/lora_dataset/train.jsonl 和 val.jsonl）：
    每行一条 JSON，messages 字段为对话轮次列表：
    {"messages": [
        {"role": "system",    "content": "你是御厨·臻享..."},
        {"role": "user",      "content": "..."},
        {"role": "assistant", "content": "..."}
    ], "category": "fridge"}

输出：
    data/lora_adapter/        — 最终 LoRA adapter 权重
    data/lora_adapter/checkpoints/   — 每 epoch 的检查点
"""

import os
import sys
import time
import logging
import json

# ---- MPS fallback（Apple Silicon 必须在 import torch 之前设置）----
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    EarlyStoppingCallback,
)
from trl import SFTTrainer, SFTConfig
from tqdm import tqdm

# ── 项目根路径 ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

# ── 日志 ─────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("lora_train")

# ── 常量 ─────────────────────────────────────────────────────────────────────
BASE_MODEL_ID   = "Qwen/Qwen2.5-7B-Instruct"
TRAIN_JSONL     = os.path.join(PROJECT_ROOT, "data", "lora_dataset", "train.jsonl")
VAL_JSONL       = os.path.join(PROJECT_ROOT, "data", "lora_dataset", "val.jsonl")
ADAPTER_DIR     = os.path.join(PROJECT_ROOT, "data", "lora_adapter")
CHECKPOINT_DIR  = os.path.join(ADAPTER_DIR, "checkpoints")
MAX_SEQ_LEN     = 1024


# ── 设备检测 ──────────────────────────────────────────────────────────────────
def detect_device() -> tuple[str, bool]:
    """
    QLoRA 只支持 CUDA（bitsandbytes 依赖 CUDA kernel）。
    返回 ("cuda", use_bf16)，无 CUDA 则抛出错误。
    """
    if torch.cuda.is_available():
        device = "cuda"
        use_bf16 = torch.cuda.is_bf16_supported()
        logger.info(f"检测到 CUDA 设备：{torch.cuda.get_device_name(0)}，bf16={use_bf16}")
        return device, use_bf16

    raise RuntimeError(
        "QLoRA（bitsandbytes 4-bit）需要 CUDA GPU。\n"
        "请在 Google Colab（T4/A100）或带 NVIDIA GPU 的机器上运行。"
    )


# ── QLoRA 4-bit 量化配置 ────────────────────────────────────────────────────────
def build_bnb_config() -> BitsAndBytesConfig:
    """NF4 4-bit 量化，双精度量化进一步节省显存。"""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


# ── 数据加载 ──────────────────────────────────────────────────────────────────
def load_jsonl(path: str) -> list[dict]:
    """逐行读取 JSONL 文件，返回 dict 列表。"""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"找不到训练数据：{path}\n"
            f"请先运行 lora_tuning/dataset_builder.py 生成训练集，"
            f"或手动创建 data/lora_dataset/ 目录并放入 train.jsonl/val.jsonl。"
        )
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    logger.info(f"已加载 {len(records)} 条数据：{path}")
    return records


# ── 数据格式化 ────────────────────────────────────────────────────────────────
def format_dataset(records: list[dict], tokenizer) -> Dataset:
    """
    将 messages 格式的记录转为 HuggingFace Dataset，
    text 字段包含 apply_chat_template 格式化后的完整对话。
    """
    texts = []
    logger.info("正在格式化训练数据...")
    for item in tqdm(records, desc="apply_chat_template", leave=False):
        messages = item.get("messages", [])
        if not messages:
            continue
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append({"text": text})
        except Exception as e:
            logger.warning(f"格式化失败，已跳过：{e}")
    logger.info(f"格式化完成，有效样本数：{len(texts)}")
    return Dataset.from_list(texts)


# ── LoRA 配置 ─────────────────────────────────────────────────────────────────
def build_lora_config() -> LoraConfig:
    return LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


# ── 训练参数 ──────────────────────────────────────────────────────────────────
def build_training_args(use_bf16: bool) -> SFTConfig:
    """
    针对 Colab T4 16GB（QLoRA 7B）优化的训练参数。
    SFTConfig 继承自 TrainingArguments，新增 max_length / dataset_text_field。
    """
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # T4 不支持 bf16，用 fp16；A100/H100 支持 bf16
    use_fp16 = not use_bf16

    return SFTConfig(
        output_dir=CHECKPOINT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=use_bf16,
        fp16=use_fp16,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,          # 最多保留 2 个 checkpoint，节省磁盘
        report_to="none",            # 不上报 wandb/tensorboard
        dataloader_num_workers=0,    # Colab 下 >0 容易死锁
        optim="paged_adamw_32bit",   # QLoRA 推荐优化器，节省显存
        # SFTConfig 专属
        max_length=MAX_SEQ_LEN,
        dataset_text_field="text",
        packing=False,               # 简单起见不做样本拼接
    )


# ── 主训练流程 ────────────────────────────────────────────────────────────────
def main():
    start_time = time.time()
    device, use_bf16 = detect_device()

    # 1. 加载 tokenizer
    logger.info(f"正在从 HuggingFace 加载 tokenizer：{BASE_MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_ID,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"   # SFT 训练时 right padding 更稳定

    # 2. 加载原始数据 & 格式化
    train_records = load_jsonl(TRAIN_JSONL)
    val_records   = load_jsonl(VAL_JSONL)
    train_dataset = format_dataset(train_records, tokenizer)
    val_dataset   = format_dataset(val_records,   tokenizer)

    # 3. 加载基础模型（4-bit QLoRA）
    logger.info(f"正在以 QLoRA 4-bit 量化加载基础模型：{BASE_MODEL_ID}")
    bnb_config = build_bnb_config()

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # QLoRA 必须先调用此函数，将量化层标记为可训练并启用 gradient checkpointing
    model = prepare_model_for_kbit_training(model)

    # 4. 注入 LoRA
    lora_config = build_lora_config()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 5. 训练参数
    training_args = build_training_args(use_bf16)

    # 6. SFTTrainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # 7. 开始训练
    logger.info("=" * 60)
    logger.info("  开始 QLoRA 微调「御厨·臻享」私厨人设 (7B)...")
    logger.info(f"  训练集：{len(train_dataset)} 条  验证集：{len(val_dataset)} 条")
    logger.info("=" * 60)

    train_result = trainer.train()

    # 8. 保存 LoRA adapter（不保存完整模型权重）
    os.makedirs(ADAPTER_DIR, exist_ok=True)
    trainer.model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)

    # 9. 打印训练摘要
    elapsed = time.time() - start_time
    final_loss = train_result.training_loss
    minutes, seconds = divmod(int(elapsed), 60)

    logger.info("\n" + "=" * 60)
    logger.info("  LoRA 微调完成！")
    logger.info(f"  训练耗时   : {minutes}m {seconds}s")
    logger.info(f"  最终 Loss  : {final_loss:.4f}")
    logger.info(f"  Adapter 路径: {ADAPTER_DIR}")
    logger.info("=" * 60)

    # 保存训练摘要 JSON
    summary = {
        "base_model":   BASE_MODEL_ID,
        "adapter_path": ADAPTER_DIR,
        "train_samples": len(train_dataset),
        "val_samples":   len(val_dataset),
        "elapsed_seconds": round(elapsed, 1),
        "final_train_loss": round(final_loss, 4),
        "device": device,
        "bf16": use_bf16,
    }
    summary_path = os.path.join(ADAPTER_DIR, "train_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info(f"训练摘要已保存至：{summary_path}")


if __name__ == "__main__":
    main()
