"""Local evaluation: PPL + KMMLU accuracy + speed."""
import os
import sys
import time
import json
import gc
import torch
import numpy as np
from pathlib import Path
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def measure_ppl(model, tokenizer, texts, max_length=512):
    total_loss = 0
    total_tokens = 0
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            outputs = model(**inputs, labels=inputs["input_ids"])
            total_loss += outputs.loss.item() * inputs["input_ids"].size(1)
            total_tokens += inputs["input_ids"].size(1)
    return np.exp(total_loss / total_tokens)


def measure_speed(model, tokenizer, prompts, max_new_tokens=64, num_runs=5):
    warmup = tokenizer("Hello", return_tensors="pt").to(model.device)
    with torch.no_grad():
        _ = model.generate(**warmup, max_new_tokens=5, do_sample=False)
    torch.cuda.synchronize()

    total_tokens = 0
    total_time = 0
    for prompt in prompts[:num_runs]:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        gen_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
        total_tokens += gen_tokens
        total_time += elapsed
    return total_tokens / total_time, total_time / total_tokens


def measure_kmmlu_accuracy(model, tokenizer, num_per_category=10, categories=None):
    """Evaluate KMMLU accuracy as proxy for competition PerfNorm."""
    if categories is None:
        categories = [
            'Computer-Science', 'Math', 'Law', 'Economics', 'Korean-History',
            'Biology', 'Chemistry', 'Psychology', 'Education', 'Management',
        ]
    answer_map = {1: 'A', 2: 'B', 3: 'C', 4: 'D'}
    correct = 0
    total = 0

    for config in categories:
        try:
            ds = load_dataset('HAERAE-HUB/KMMLU', config, split='test')
            items = list(ds.select(range(min(num_per_category, len(ds)))))
            for item in items:
                prompt = (
                    f"다음 문제의 정답을 선택하세요.\n\n"
                    f"{item['question']}\n"
                    f"A: {item['A']}\nB: {item['B']}\nC: {item['C']}\nD: {item['D']}\n\n"
                    f"정답:"
                )
                conversation = [{"role": "user", "content": prompt}]
                text = tokenizer.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs, max_new_tokens=10, do_sample=False,
                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    )
                gen_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
                correct_letter = answer_map.get(item['answer'], 'A')
                if correct_letter in gen_text[:5]:
                    correct += 1
                total += 1
        except Exception as e:
            print(f"  [WARN] {config}: {e}")

    accuracy = correct / total if total > 0 else 0
    return accuracy, correct, total


def evaluate_model(model_path, model_name, base_results=None):
    """Full evaluation of a model."""
    clear_gpu()
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*60}")

    model_size = sum(f.stat().st_size for f in Path(model_path).glob("*.safetensors")) / (1024**3)
    print(f"Size: {model_size:.2f} GB")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto",
    )
    model.eval()

    gpu_mem = torch.cuda.max_memory_allocated() / (1024**3)
    print(f"GPU Memory: {gpu_mem:.2f} GB")

    # PPL
    print("Measuring PPL...")
    ds = load_dataset("LGAI-EXAONE/MANTA-1M", split="train[:40]")
    eval_texts = []
    for item in ds:
        if "conversations" in item:
            text = " ".join([c.get("content", "") for c in item["conversations"] if c.get("content")])
            if len(text) > 100:
                eval_texts.append(text[:800])
    eval_texts = eval_texts[:20]
    ppl = measure_ppl(model, tokenizer, eval_texts)
    print(f"PPL: {ppl:.2f}")

    # Speed
    print("Measuring speed...")
    speed_prompts = [
        "한국의 수도는 어디인가요?",
        "인공지능이 미래 사회에 미치는 영향을 설명해주세요.",
        "Python과 Java의 차이점은 무엇인가요?",
        "기후 변화의 원인과 해결책을 알려주세요.",
        "효과적인 학습 방법에 대해 조언해주세요.",
    ]
    tps, tpt = measure_speed(model, tokenizer, speed_prompts)
    print(f"Speed: {tps:.2f} tok/s, {tpt*1000:.2f} ms/tok")

    # KMMLU Accuracy
    print("Measuring KMMLU accuracy...")
    kmmlu_acc, kmmlu_correct, kmmlu_total = measure_kmmlu_accuracy(model, tokenizer, num_per_category=10)
    print(f"KMMLU Accuracy: {kmmlu_acc:.4f} ({kmmlu_correct}/{kmmlu_total})")

    del model, tokenizer
    clear_gpu()

    result = {
        "name": model_name,
        "path": model_path,
        "size_gb": model_size,
        "gpu_mem_gb": gpu_mem,
        "perplexity": ppl,
        "tokens_per_sec": tps,
        "time_per_token": tpt,
        "kmmlu_accuracy": kmmlu_acc,
        "kmmlu_correct": kmmlu_correct,
        "kmmlu_total": kmmlu_total,
    }

    if base_results:
        perf_norm = base_results["perplexity"] / ppl
        speed_norm = 1 - (tpt / base_results["time_per_token"])
        score = max(0.5 * perf_norm + 0.5 * speed_norm, 0)
        result["perf_norm"] = perf_norm
        result["speed_norm"] = speed_norm
        result["local_score"] = score
        print(f"PerfNorm: {perf_norm:.4f}, SpeedNorm: {speed_norm:.4f}, Score: {score:.4f}")

    return result


if __name__ == "__main__":
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)

    model_path = sys.argv[1] if len(sys.argv) > 1 else "./base_model"
    model_name = sys.argv[2] if len(sys.argv) > 2 else os.path.basename(model_path)

    if torch.cuda.is_available():
        torch.cuda.init()
        torch.zeros(1).cuda()
        print(f"[INFO] CUDA: {torch.cuda.get_device_name(0)}")

    # Load or compute base results
    base_results_file = os.path.join(results_dir, "base_results.json")
    base_results = None
    if os.path.exists(base_results_file):
        with open(base_results_file) as f:
            base_results = json.load(f)
        print(f"[INFO] Using cached base results: PPL={base_results['perplexity']:.2f}")

    if model_path != "./base_model" and model_path != "/home/lgaimers/base_model":
        result = evaluate_model(model_path, model_name, base_results)
    else:
        result = evaluate_model(model_path, model_name)
        with open(base_results_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[INFO] Base results saved to {base_results_file}")

    output_file = os.path.join(results_dir, f"{model_name}.json")
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved: {output_file}")
