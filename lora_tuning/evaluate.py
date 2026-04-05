"""
御厨·臻享 LoRA 微调评估脚本
========================================
指标：私厨建议准确率
  - 对 test.jsonl 里每条问题，让两个模型各自生成回答
  - 用 qwen-max-latest 作为裁判，二元判断回答是否达到米其林私厨专业水准（是/否）
  - 准确率 = "是" 的条数 / 总条数 × 100%
  - 对比 base Qwen2.5-7B 与 QLoRA 微调后模型的准确率，输出提升幅度

运行方式：
    python evaluate.py            # 完整评估（base + fine-tuned）
    python evaluate.py --skip-base  # 只评估微调后模型（节省时间）

环境变量：
    DASHSCOPE_API_KEY  — qwen-max-latest 裁判调用必需
"""

import os
import sys
import json
import time
import logging
import argparse

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
from tqdm import tqdm
from openai import OpenAI
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ── 项目根路径 ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("lora_eval")

# ── 常量 ───────────────────────────────────────────────────────────────────────
BASE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER_DIR   = os.path.join(PROJECT_ROOT, "data", "lora_adapter")
TEST_JSONL    = os.path.join(PROJECT_ROOT, "data", "lora_dataset", "test.jsonl")
REPORT_PATH   = os.path.join(PROJECT_ROOT, "data", "lora_eval_report.json")

GEN_KWARGS = dict(
    max_new_tokens=200,
    temperature=0.7,
    do_sample=True,
    pad_token_id=None,
)

# 私厨建议准确率裁判 prompt
JUDGE_PROMPT_TMPL = """你是一位米其林餐厅评审员。请判断以下 AI 私厨助手的回答是否达到高端私厨服务标准。

评判维度（以下四点综合考量，不要求全部完美，但需大多达标）：
1. 信息准确：烹饪/营养/食品安全知识基本正确，无明显错误
2. 建议具体：给出了有实际帮助的具体操作指导，而非模糊泛泛的建议
3. 专业深度：体现出超越普通家庭厨师的专业烹饪知识
4. 私厨人设：语气得体，体现专业服务感（区别于通用 AI 的中性语气）

用户问题：{question}
AI回答：{answer}

只输出：是 或 否，不要任何解释。"""

# 食品安全专项裁判 prompt
SAFETY_JUDGE_TMPL = """你是一位食品安全专家。请判断以下 AI 助手对食品安全问题的回答是否正确且有益。

评判标准（综合考量）：
1. 正确识别了食品安全危险（变质、有毒、有害成分等）
2. 给出了恰当的处置建议（如停止食用、就医、漱口、避免食用、冷藏处理等，具体建议视场景而定）
3. 回答不会引导用户做出危险行为

用户问题：{question}
AI回答：{answer}

只输出：是 或 否，不要任何解释。"""


# ── 设备检测 ───────────────────────────────────────────────────────────────────
def detect_device() -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        logger.info(f"使用 CUDA：{torch.cuda.get_device_name(0)}")
        return "cuda", torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("使用 Apple Silicon MPS")
        return "mps", torch.bfloat16
    logger.warning("使用 CPU（推理速度较慢）")
    return "cpu", torch.float32


# ── 数据加载 ───────────────────────────────────────────────────────────────────
def load_test_data(path: str) -> list[dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"找不到测试集：{path}\n"
            f"请先运行 dataset_builder.py 生成数据集。"
        )
    # 修正旧版本中过于严格的 correct_action 关键词
    KEYWORD_FIXES = {"不要冲洗": "不要", "吐掉": "吐"}

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                record = json.loads(line)
                if record.get("correct_action") in KEYWORD_FIXES:
                    record["correct_action"] = KEYWORD_FIXES[record["correct_action"]]
                records.append(record)
    logger.info(f"已加载测试集：{len(records)} 条")
    return records


# ── QLoRA 4-bit 量化配置（推理复用，节省显存）─────────────────────────────────
def _bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


# ── 模型加载 ───────────────────────────────────────────────────────────────────
def load_base_model(device: str, torch_dtype: torch.dtype):
    logger.info(f"加载 base 模型（4-bit QLoRA）：{BASE_MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=_bnb_config(),
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def load_finetuned_model(device: str, torch_dtype: torch.dtype):
    if not os.path.exists(ADAPTER_DIR):
        raise FileNotFoundError(
            f"找不到 LoRA adapter：{ADAPTER_DIR}\n"
            f"请先运行 train_lora.py 完成训练。"
        )
    logger.info(f"加载 fine-tuned 模型（4-bit QLoRA + adapter）：{BASE_MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=_bnb_config(),
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model.eval()
    return model, tokenizer


# ── 推理 ───────────────────────────────────────────────────────────────────────
@torch.inference_mode()
def generate_response(model, tokenizer, messages: list[dict], device: str) -> str:
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    gen_kwargs = {**GEN_KWARGS, "pad_token_id": tokenizer.eos_token_id}
    output_ids = model.generate(**inputs, **gen_kwargs)
    new_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


# ── 裁判：qwen-max-latest 二元判断 ────────────────────────────────────────────────────
def build_judge_client() -> OpenAI | None:
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        logger.warning("未检测到 DASHSCOPE_API_KEY，私厨建议准确率将无法计算。")
        return None
    return OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )


# ── 裁判 API 通用调用 ─────────────────────────────────────────────────────────
def _call_judge(prompt: str, client: OpenAI) -> bool | None:
    """调用 qwen-max-latest 做二元判断，返回 True/False/None（失败时）。"""
    try:
        response = client.chat.completions.create(
            model="qwen-max-latest",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.0,
        )
        raw = response.choices[0].message.content.strip()
        return "是" in raw
    except Exception as e:
        logger.debug(f"Judge API 调用失败：{e}")
        return None


# ── 指标一：私厨建议准确率（LLM-as-Judge）────────────────────────────────────────
def judge_chef_accuracy(question: str, answer: str, client: OpenAI) -> bool | None:
    """用 qwen-max-latest 判断私厨回答质量，返回 True/False/None。"""
    prompt = JUDGE_PROMPT_TMPL.format(question=question, answer=answer)
    return _call_judge(prompt, client)


# ── 指标二：食品安全预警准确率（LLM-as-Judge，取代脆弱的关键词匹配）──────────────
def judge_food_safety(question: str, answer: str, client: OpenAI) -> bool | None:
    """用 qwen-max-latest 判断食品安全回答是否正确有益，返回 True/False/None。"""
    prompt = SAFETY_JUDGE_TMPL.format(question=question, answer=answer)
    return _call_judge(prompt, client)


def compute_food_safety_accuracy(results: list[dict]) -> float | None:
    """
    针对 category='safety' 的条目统计 LLM 裁判通过率。
    读取 evaluate_model 中已存储的 safety_correct 字段。
    """
    safety = [r for r in results if r.get("category") == "safety"
              and r.get("safety_correct") is not None]
    if not safety:
        return None
    correct = sum(1 for r in safety if r["safety_correct"])
    return correct / len(safety) * 100.0


# ── 核心评估 ───────────────────────────────────────────────────────────────────
def evaluate_model(
    model,
    tokenizer,
    test_data: list[dict],
    model_name: str,
    device: str,
    judge_client: OpenAI | None,
) -> dict:
    """
    生成回答，计算两个指标：
    - 私厨建议准确率（qwen-max-latest 裁判，全量 100 条）
    - 食品安全预警准确率（客观关键词匹配，仅 safety 类 ~25 条）
    """
    results = []
    logger.info(f"\n开始评估：{model_name}（共 {len(test_data)} 条）")

    for item in tqdm(test_data, desc=f"评估 {model_name}", unit="条"):
        messages       = item.get("messages", [])
        category       = item.get("category", "general")
        correct_action = item.get("correct_action")   # 仅 safety 类有

        user_msgs = [m for m in messages if m.get("role") == "user"]
        question  = user_msgs[-1]["content"] if user_msgs else ""

        prompt_messages = [m for m in messages if m.get("role") != "assistant"]
        answer = generate_response(model, tokenizer, prompt_messages, device)

        # 指标一：私厨建议准确率（LLM 裁判，全量）
        judge_correct = judge_chef_accuracy(question, answer, judge_client) if judge_client else None

        # 指标二：食品安全预警准确率（LLM 裁判，仅 safety 类）
        safety_correct = None
        if category == "safety" and judge_client:
            safety_correct = judge_food_safety(question, answer, judge_client)

        results.append({
            "model":          model_name,
            "category":       category,
            "question":       question,
            "answer":         answer,
            "judge_correct":  judge_correct,
            "safety_correct": safety_correct,
            "correct_action": correct_action,
        })

    # 指标一：私厨建议准确率
    judged = [r for r in results if r["judge_correct"] is not None]
    chef_accuracy = (
        sum(1 for r in judged if r["judge_correct"]) / len(judged) * 100.0
        if judged else None
    )

    # 指标二：食品安全预警准确率
    food_safety_accuracy = compute_food_safety_accuracy(results)

    return {
        "model_name":               model_name,
        "sample_count":             len(results),
        "chef_accuracy_pct":        chef_accuracy,
        "food_safety_accuracy_pct": food_safety_accuracy,
        "detail":                   results,
    }


# ── 报告打印 & 保存 ─────────────────────────────────────────────────────────────
def print_report(base_metrics: dict | None, ft_metrics: dict, elapsed: float):
    def fmt(val):
        return f"{val:.1f}%" if val is not None else "N/A"

    def delta_str(base_val, ft_val):
        if base_val is None or ft_val is None:
            return "N/A"
        d = ft_val - base_val
        return f"{'↑' if d >= 0 else '↓'}{abs(d):.1f}pp"

    print()
    print("╔════════════════════════════════════════════════════════╗")
    print("║           御厨·臻享 LoRA 微调评估报告                   ║")
    print("╚════════════════════════════════════════════════════════╝")
    print()

    ft_chef  = ft_metrics.get("chef_accuracy_pct")
    ft_safe  = ft_metrics.get("food_safety_accuracy_pct")

    if base_metrics:
        base_chef = base_metrics.get("chef_accuracy_pct")
        base_safe = base_metrics.get("food_safety_accuracy_pct")

        print("┌──────────────────────────┬────────────┬────────────┬──────────┐")
        print("│ 指标                     │ Base 模型  │ 微调后模型 │  提升    │")
        print("├──────────────────────────┼────────────┼────────────┼──────────┤")
        print(f"│ 私厨建议准确率（LLM裁判） │ {fmt(base_chef):^10} │ {fmt(ft_chef):^10} │ {delta_str(base_chef, ft_chef):^8} │")
        print(f"│ 食品安全预警准确率（LLM裁判）│ {fmt(base_safe):^10} │ {fmt(ft_safe):^10} │ {delta_str(base_safe, ft_safe):^8} │")
        print("└──────────────────────────┴────────────┴────────────┴──────────┘")
    else:
        print("┌──────────────────────────┬────────────┐")
        print("│ 指标                     │ 微调后模型 │")
        print("├──────────────────────────┼────────────┤")
        print(f"│ 私厨建议准确率（LLM裁判） │ {fmt(ft_chef):^10} │")
        print(f"│ 食品安全预警准确率（客观）│ {fmt(ft_safe):^10} │")
        print("└──────────────────────────┴────────────┘")

    print()
    minutes, seconds = divmod(int(elapsed), 60)
    print(f"评估样本数：{ft_metrics['sample_count']} 条  |  裁判模型：qwen-max-latest  |  用时：{minutes}m {seconds}s")
    print(f"结果已保存至：{REPORT_PATH}")
    print()


def save_report(base_metrics: dict | None, ft_metrics: dict, elapsed: float):
    report = {
        "judge_model":         "qwen-max-latest",
        "elapsed_seconds":     round(elapsed, 1),
        "base_model":          base_metrics,
        "finetuned_model":     ft_metrics,
        "report_generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"完整报告已写入：{REPORT_PATH}")


# ── 入口 ───────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="御厨·臻享 LoRA 微调评估")
    parser.add_argument(
        "--skip-base", action="store_true", default=False,
        help="跳过 base 模型评估，只评估 fine-tuned 模型（节省时间）",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time.time()
    device, torch_dtype = detect_device()

    test_data = load_test_data(TEST_JSONL)
    judge_client = build_judge_client()

    base_metrics = None

    if not args.skip_base:
        logger.info("第一阶段：评估 base 模型...")
        base_model, base_tokenizer = load_base_model(device, torch_dtype)
        base_metrics = evaluate_model(
            base_model, base_tokenizer, test_data,
            model_name="Qwen2.5-7B-Instruct (base)",
            device=device,
            judge_client=judge_client,
        )
        del base_model
        torch.cuda.empty_cache()
        logger.info("base 模型评估完成，已释放显存。")
    else:
        logger.info("已跳过 base 模型评估（--skip-base）。")

    logger.info("第二阶段：评估 fine-tuned 模型（base + LoRA adapter）...")
    ft_model, ft_tokenizer = load_finetuned_model(device, torch_dtype)
    ft_metrics = evaluate_model(
        ft_model, ft_tokenizer, test_data,
        model_name="Qwen2.5-7B-Instruct + QLoRA (御厨·臻享)",
        device=device,
        judge_client=judge_client,
    )

    elapsed = time.time() - start_time
    print_report(base_metrics, ft_metrics, elapsed)
    save_report(base_metrics, ft_metrics, elapsed)


if __name__ == "__main__":
    main()
