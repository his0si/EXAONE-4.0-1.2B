#!/usr/bin/env python3
"""
Test 3 scenarios to reproduce server behavior differences.

Scenario 1: Truncate prompts to 512 tokens (simulating tokenizer max_length=512)
Scenario 2: max_model_len=65536 (server default, no explicit limit)
Scenario 3: No prompt token limit (server might not truncate prompts)

Baseline: Current local settings (max_model_len=16384, max_prompt_tokens=15360)
"""
import os, sys, time, json, re, argparse
import warnings
warnings.filterwarnings("ignore")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

# All functions are inlined below


def load_tokenizer(model_path):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)


def load_ko_longrag():
    from datasets import load_dataset
    ds = load_dataset("LGAI-EXAONE/Ko-LongRAG", split="test")
    cols = ds.column_names
    samples = []
    for row in ds:
        context = str(row.get("context", ""))
        question = str(row.get("question", ""))
        answer = row.get("answer", "")
        if isinstance(answer, list):
            answer = answer[0] if answer else ""
        answer = str(answer)
        raw_prompt = str(row.get("prompt", ""))
        titles_raw = row.get("titles", "")
        if isinstance(titles_raw, list):
            titles_str = ", ".join(str(t) for t in titles_raw)
        else:
            titles_str = str(titles_raw)
        samples.append({
            "context": context, "question": question,
            "answer": answer, "raw_prompt": raw_prompt,
            "titles": titles_str,
        })
    return samples


def load_kmmlu_pro():
    """Load a subset of KMMLU-Pro for quick testing."""
    from datasets import load_dataset
    ds = load_dataset("LGAI-EXAONE/KMMLU-Pro", split="test")
    samples = []
    for row in ds:
        question = str(row.get("question", ""))
        choices = row.get("options", row.get("choices", []))
        if isinstance(choices, dict):
            choices = list(choices.values())
        answer_raw = row.get("solution", row.get("answer", ""))
        answer = ""
        if isinstance(answer_raw, int):
            answer = chr(ord("A") + answer_raw)
        elif isinstance(answer_raw, str):
            s = answer_raw.strip()
            if s.isdigit():
                idx = int(s)
                if idx >= 1 and choices and idx <= len(choices):
                    answer = chr(ord("A") + idx - 1)
                else:
                    answer = chr(ord("A") + idx)
            elif s and s[0].upper() in "ABCDE":
                answer = s[0].upper()
        samples.append({"question": question, "choices": choices, "answer": answer})
    return samples


def build_qa_content(sample, context_override=None):
    ctx = context_override if context_override is not None else sample["context"]
    if sample.get("raw_prompt") and sample["context"]:
        content = sample["raw_prompt"]
        content = content.replace("{titles}", sample.get("titles", ""))
        content = content.replace("{context}", ctx)
        content = content.replace("{question}", sample.get("question", ""))
    elif ctx:
        content = (
            f"다음 문서를 읽고 질문에 답하세요.\n\n"
            f"문서:\n{ctx}\n\n"
            f"질문: {sample['question']}\n\n"
            f"답변:"
        )
    else:
        content = f"질문: {sample['question']}\n답변:"
    return content


def format_qa_prompt(sample, tokenizer, max_prompt_tokens=None):
    content = build_qa_content(sample)
    messages = [{"role": "user", "content": content}]
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False)

    if max_prompt_tokens is not None:
        token_ids = tokenizer.encode(prompt)
        if len(token_ids) > max_prompt_tokens:
            context = sample["context"]
            lo, hi = 0, len(context)
            while lo < hi:
                mid = (lo + hi + 1) // 2
                test_content = build_qa_content(sample, context_override=context[:mid])
                test_msgs = [{"role": "user", "content": test_content}]
                test_prompt = tokenizer.apply_chat_template(
                    test_msgs, add_generation_prompt=True, tokenize=False)
                if len(tokenizer.encode(test_prompt)) <= max_prompt_tokens:
                    lo = mid
                else:
                    hi = mid - 1
            content = build_qa_content(sample, context_override=context[:lo])
            messages = [{"role": "user", "content": content}]
            prompt = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False)
    return prompt


def format_mcqa_prompt(sample, tokenizer):
    labels = "ABCDE"
    choice_text = "\n".join(
        f"{labels[i]}. {c}" for i, c in enumerate(sample["choices"])
    ) if sample["choices"] else ""
    content = f"{sample['question']}\n{choice_text}\n정답:"
    messages = [{"role": "user", "content": content}]
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False)


def truncate_prompt_to_tokens(prompt, tokenizer, max_tokens):
    """Truncate prompt at token level."""
    ids = tokenizer.encode(prompt)
    if len(ids) <= max_tokens:
        return prompt
    truncated_ids = ids[:max_tokens]
    return tokenizer.decode(truncated_ids, skip_special_tokens=False)


def strip_thinking(text):
    stripped = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    if stripped.startswith('</think>'):
        stripped = stripped[len('</think>'):].strip()
    return stripped


def compute_f1(prediction, reference):
    pred_clean = re.sub(r'[^\w\s가-힣]', ' ', prediction.lower()).strip()
    ref_clean = re.sub(r'[^\w\s가-힣]', ' ', reference.lower()).strip()
    pred_tokens = pred_clean.split()
    ref_tokens = ref_clean.split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = set(pred_tokens) & set(ref_tokens)
    if not common:
        pred_chars = list(pred_clean.replace(" ", ""))
        ref_chars = list(ref_clean.replace(" ", ""))
        common_chars = set(pred_chars) & set(ref_chars)
        if not common_chars or not pred_chars or not ref_chars:
            return 0.0
        precision = len(common_chars) / len(pred_chars)
        recall = len(common_chars) / len(ref_chars)
        return 2 * precision * recall / (precision + recall)
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def parse_mcqa_answer(text):
    text = strip_thinking(text).strip()
    text_upper = text.upper()
    m = re.search(r'정답[은:]?\s*\**\s*([A-Ea-e])', text)
    if m: return m.group(1).upper()
    m = re.search(r'[Aa]nswer[:\s]+\**\s*([A-Ea-e])', text)
    if m: return m.group(1).upper()
    first_line = text_upper.split('\n')[0].strip()
    if first_line and first_line[0] in "ABCDE":
        return first_line[0]
    m = re.search(r'[(\s]([A-E])[)\s.,]', text_upper)
    if m: return m.group(1)
    for ch in first_line:
        if ch in "ABCDE": return ch
    for ch in text_upper:
        if ch in "ABCDE": return ch
    return ""


def run_qa_eval(llm, prompts, references, sampling):
    """Run QA evaluation and return F1 score."""
    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sampling)
    gen_time = time.perf_counter() - t0

    predictions = []
    gen_tokens = 0
    for o in outputs:
        text = strip_thinking(o.outputs[0].text).strip()
        first_line = text.split("\n")[0].strip()
        gen_tokens += len(o.outputs[0].token_ids)
        predictions.append(first_line)

    scores = [compute_f1(p, r) for p, r in zip(predictions, references)]
    f1 = sum(scores) / len(scores)
    return f1, gen_time, gen_tokens, predictions


def run_mcqa_eval(llm, prompts, references, sampling):
    """Run MCQA evaluation and return accuracy."""
    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sampling)
    gen_time = time.perf_counter() - t0

    predictions = []
    gen_tokens = 0
    for o in outputs:
        text = o.outputs[0].text
        gen_tokens += len(o.outputs[0].token_ids)
        predictions.append(parse_mcqa_answer(text))

    correct = sum(1 for p, r in zip(predictions, references) if p == r)
    accuracy = correct / len(predictions)
    return accuracy, gen_time, gen_tokens, predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--benchmarks", default="ko_longrag,kmmlu_pro",
                        help="Comma-separated benchmarks to test")
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    model_path = args.model
    tokenizer = load_tokenizer(model_path)
    benchmarks = args.benchmarks.split(",")

    print(f"\n{'='*70}")
    print(f"  Model: {model_path}")
    print(f"  max_model_len: {args.max_model_len}")
    print(f"  Benchmarks: {benchmarks}")
    print(f"{'='*70}")

    # Load vLLM
    print(f"\n[1/5] Loading vLLM (max_model_len={args.max_model_len})...")
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.85,
        max_model_len=args.max_model_len,
    )

    sampling_qa = SamplingParams(temperature=0.0, max_tokens=16384)
    sampling_mcqa = SamplingParams(temperature=0.0, max_tokens=512)

    # Load benchmarks
    print("[2/5] Loading benchmarks...")
    qa_samples = load_ko_longrag() if "ko_longrag" in benchmarks else None
    mcqa_samples = load_kmmlu_pro() if "kmmlu_pro" in benchmarks else None

    results = {}

    # ═══════════════════════════════════════════════════════════════
    # Baseline: current local settings (max_prompt_tokens = max_model_len - 1024)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("  [BASELINE] max_prompt_tokens = {args.max_model_len} - 1024")
    print(f"{'='*70}")

    max_pt_baseline = args.max_model_len - 1024

    if qa_samples:
        prompts = [format_qa_prompt(s, tokenizer, max_prompt_tokens=max_pt_baseline) for s in qa_samples]
        refs = [s["answer"] for s in qa_samples]
        tok_lens = [len(tokenizer.encode(p)) for p in prompts]
        print(f"  ko_longrag: {len(prompts)} samples, prompt tokens: min={min(tok_lens)}, max={max(tok_lens)}, avg={sum(tok_lens)/len(tok_lens):.0f}")
        f1, t, tokens, preds = run_qa_eval(llm, prompts, refs, sampling_qa)
        print(f"  ko_longrag F1={f1:.4f} | time={t:.1f}s | tokens={tokens}")
        results["baseline_ko_longrag"] = f1

    if mcqa_samples:
        prompts = [format_mcqa_prompt(s, tokenizer) for s in mcqa_samples]
        refs = [s["answer"] for s in mcqa_samples]
        acc, t, tokens, preds = run_mcqa_eval(llm, prompts, refs, sampling_mcqa)
        print(f"  kmmlu_pro  Acc={acc:.4f} | time={t:.1f}s | tokens={tokens}")
        results["baseline_kmmlu_pro"] = acc

    # ═══════════════════════════════════════════════════════════════
    # Scenario 1: Truncate ALL prompts to 512 tokens
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("  [SCENARIO 1] Truncate ALL prompts to 512 tokens")
    print(f"{'='*70}")

    if qa_samples:
        prompts_full = [format_qa_prompt(s, tokenizer, max_prompt_tokens=None) for s in qa_samples]
        prompts = [truncate_prompt_to_tokens(p, tokenizer, 512) for p in prompts_full]
        refs = [s["answer"] for s in qa_samples]
        tok_lens = [len(tokenizer.encode(p)) for p in prompts]
        print(f"  ko_longrag: {len(prompts)} samples, prompt tokens: min={min(tok_lens)}, max={max(tok_lens)}, avg={sum(tok_lens)/len(tok_lens):.0f}")
        f1, t, tokens, preds = run_qa_eval(llm, prompts, refs, sampling_qa)
        print(f"  ko_longrag F1={f1:.4f} | time={t:.1f}s | tokens={tokens}")
        results["s1_trunc512_ko_longrag"] = f1

    if mcqa_samples:
        prompts_full = [format_mcqa_prompt(s, tokenizer) for s in mcqa_samples]
        prompts = [truncate_prompt_to_tokens(p, tokenizer, 512) for p in prompts_full]
        refs = [s["answer"] for s in mcqa_samples]
        acc, t, tokens, preds = run_mcqa_eval(llm, prompts, refs, sampling_mcqa)
        print(f"  kmmlu_pro  Acc={acc:.4f} | time={t:.1f}s | tokens={tokens}")
        results["s1_trunc512_kmmlu_pro"] = acc

    # ═══════════════════════════════════════════════════════════════
    # Scenario 2: No prompt token limit (let prompts be full length)
    # This tests if the server doesn't truncate prompts at all
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("  [SCENARIO 2] No prompt token limit (full length prompts)")
    print(f"{'='*70}")

    if qa_samples:
        prompts = [format_qa_prompt(s, tokenizer, max_prompt_tokens=None) for s in qa_samples]
        refs = [s["answer"] for s in qa_samples]
        tok_lens = [len(tokenizer.encode(p)) for p in prompts]
        print(f"  ko_longrag: {len(prompts)} samples, prompt tokens: min={min(tok_lens)}, max={max(tok_lens)}, avg={sum(tok_lens)/len(tok_lens):.0f}")

        # Check if any exceed max_model_len - need to skip those or they'll error
        over_limit = sum(1 for l in tok_lens if l > args.max_model_len - 256)
        if over_limit:
            print(f"  WARNING: {over_limit} prompts exceed max_model_len-256={args.max_model_len-256}")
            print(f"  Truncating those to max_model_len - 256")
            safe_limit = args.max_model_len - 256
            prompts = [
                truncate_prompt_to_tokens(p, tokenizer, safe_limit) if len(tokenizer.encode(p)) > safe_limit else p
                for p in prompts
            ]
            tok_lens = [len(tokenizer.encode(p)) for p in prompts]
            print(f"  After fix: min={min(tok_lens)}, max={max(tok_lens)}, avg={sum(tok_lens)/len(tok_lens):.0f}")

        f1, t, tokens, preds = run_qa_eval(llm, prompts, refs, sampling_qa)
        print(f"  ko_longrag F1={f1:.4f} | time={t:.1f}s | tokens={tokens}")
        results["s2_nolimit_ko_longrag"] = f1

    if mcqa_samples:
        # MCQA prompts are short, no change expected
        prompts = [format_mcqa_prompt(s, tokenizer) for s in mcqa_samples]
        refs = [s["answer"] for s in mcqa_samples]
        acc, t, tokens, preds = run_mcqa_eval(llm, prompts, refs, sampling_mcqa)
        print(f"  kmmlu_pro  Acc={acc:.4f} | time={t:.1f}s | tokens={tokens}")
        results["s2_nolimit_kmmlu_pro"] = acc

    # ═══════════════════════════════════════════════════════════════
    # Scenario 3: max_gen_toks=16384 for ALL benchmarks (including MCQA)
    # Server uses max_gen_toks=16384 globally
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("  [SCENARIO 3] max_gen_toks=16384 for ALL benchmarks (incl. MCQA)")
    print(f"{'='*70}")

    sampling_all_16k = SamplingParams(temperature=0.0, max_tokens=16384)

    if qa_samples:
        # QA already uses 16384, same as baseline
        print(f"  ko_longrag: same as baseline (already uses max_tokens=16384)")
        results["s3_16k_ko_longrag"] = results.get("baseline_ko_longrag")

    if mcqa_samples:
        prompts = [format_mcqa_prompt(s, tokenizer) for s in mcqa_samples]
        refs = [s["answer"] for s in mcqa_samples]
        print(f"  kmmlu_pro: running with max_tokens=16384 (was 512)...")
        acc, t, tokens, preds = run_mcqa_eval(llm, prompts, refs, sampling_all_16k)
        print(f"  kmmlu_pro  Acc={acc:.4f} | time={t:.1f}s | tokens={tokens}")
        # Show some examples where answer differs
        results["s3_16k_kmmlu_pro"] = acc

    # ═══════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    print(f"{'Scenario':<45} {'ko_longrag':>10} {'kmmlu_pro':>10}")
    print(f"{'-'*70}")

    def fmt(v):
        return f"{v:.4f}" if v is not None else "N/A"

    print(f"{'Baseline (local settings)':<45} {fmt(results.get('baseline_ko_longrag')):>10} {fmt(results.get('baseline_kmmlu_pro')):>10}")
    print(f"{'S1: Truncate to 512 tokens':<45} {fmt(results.get('s1_trunc512_ko_longrag')):>10} {fmt(results.get('s1_trunc512_kmmlu_pro')):>10}")
    print(f"{'S2: No prompt limit':<45} {fmt(results.get('s2_nolimit_ko_longrag')):>10} {fmt(results.get('s2_nolimit_kmmlu_pro')):>10}")
    print(f"{'S3: max_gen_toks=16384 for all':<45} {fmt(results.get('s3_16k_ko_longrag')):>10} {fmt(results.get('s3_16k_kmmlu_pro')):>10}")

    # Save
    out_path = os.path.join(ROOT, "results", "server_scenario_test.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
