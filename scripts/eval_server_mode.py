#!/usr/bin/env python3
"""
Evaluate model via vLLM OpenAI API server — matches competition server.
Uses concurrent requests for realistic throughput measurement.

Fixes vs v1:
  - MCQA: detect 'solution' column (KMMLU), 1-indexed answer conversion
  - QA: token-level context truncation (prevents 400 errors)
  - Judge: handle 'turns' column (KoMT-Bench)
  - Prompt format matches 04_eval_vllm.py exactly
"""
import os, sys, re, json, time, subprocess, signal, argparse, requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def wait_for_server(base_url, timeout=300):
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{base_url}/health", timeout=5)
            if r.status_code == 200:
                return True
        except:
            pass
        time.sleep(2)
    return False


def chat_completion(base_url, messages, max_tokens=None, temperature=0.0):
    """Send a single chat completion request. If max_tokens is None, omit it
    so vLLM auto-fills with (max_model_len - prompt_tokens)."""
    body = {"model": "model", "messages": messages, "temperature": temperature}
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    r = requests.post(
        f"{base_url}/v1/chat/completions", json=body, timeout=600,
    )
    r.raise_for_status()
    data = r.json()
    text = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    return text, usage


def batch_chat(base_url, messages_list, max_tokens=None, max_workers=16):
    """Send multiple chat requests concurrently. Returns list of (text, usage) in order.
    max_tokens=None → let vLLM auto-fill (max_model_len - prompt_tokens)."""
    results = [None] * len(messages_list)

    def _call(idx, messages):
        text, usage = chat_completion(base_url, messages, max_tokens=max_tokens)
        return idx, text, usage

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = []
        for i, msgs in enumerate(messages_list):
            futures.append(pool.submit(_call, i, msgs))
        for fut in as_completed(futures):
            try:
                idx, text, usage = fut.result()
                results[idx] = (text, usage)
            except Exception as e:
                # Find the index from the future
                print(f"  [ERROR] Request failed: {e}")
    # Fill None entries with empty results
    for i in range(len(results)):
        if results[i] is None:
            results[i] = ("", {})
    return results


def load_dataset_flex(hf_id):
    from datasets import load_dataset
    for split in ["test", "validation", "train"]:
        try:
            return load_dataset(hf_id, split=split)
        except:
            continue
    info = load_dataset(hf_id)
    return info[list(info.keys())[0]]


# ════════════════════════════════════════════════════════════
#  MCQA — matches 04_eval_vllm.py exactly
# ════════════════════════════════════════════════════════════

def _detect_col(ds, candidates):
    for c in candidates:
        if c in ds.column_names:
            return c
    return None


def load_mcqa(hf_id):
    """Load MCQA benchmark with proper column detection."""
    ds = load_dataset_flex(hf_id)

    q_col = _detect_col(ds, ["question", "query", "input", "prompt", "problem"])
    c_col = _detect_col(ds, ["choices", "options", "candidates", "choice"])
    a_col = _detect_col(ds, ["answer", "label", "correct", "answer_key", "gold", "solution"])

    print(f"  Columns: {ds.column_names}")
    print(f"  Detected: q={q_col}, c={c_col}, a={a_col}")

    samples = []
    for row in ds:
        question = str(row[q_col]) if q_col else ""
        choices = row.get(c_col, []) if c_col else []
        answer_raw = row.get(a_col, "") if a_col else ""

        if isinstance(choices, dict):
            choices = list(choices.values())
        elif isinstance(choices, str):
            choices = [c.strip() for c in choices.split("\n") if c.strip()]

        # Normalize answer (matches 04_eval_vllm.py logic)
        answer = ""
        if isinstance(answer_raw, int):
            answer = chr(ord("A") + answer_raw)
        elif isinstance(answer_raw, str):
            s = answer_raw.strip()
            if s.isdigit():
                idx = int(s)
                if idx >= 1 and choices and idx <= len(choices):
                    answer = chr(ord("A") + idx - 1)  # 1-indexed → A-E
                elif idx == 0:
                    answer = "A"
                else:
                    answer = chr(ord("A") + min(idx, len(choices) - 1))
            elif s and s[0].upper() in "ABCDE":
                answer = s[0].upper()
            else:
                answer = s.upper()

        samples.append({"question": question, "choices": choices, "answer": answer})

    from collections import Counter
    ans_dist = Counter(s["answer"] for s in samples)
    print(f"  Answer distribution: {dict(ans_dist)}")
    return samples


def format_mcqa_content(sample):
    """Format MCQA user message content — matches 04_eval_vllm.py."""
    labels = "ABCDE"
    choice_text = "\n".join(
        f"{labels[i]}. {c}" for i, c in enumerate(sample["choices"])
    ) if sample["choices"] else ""
    return f"{sample['question']}\n{choice_text}\n정답:"


def parse_mcqa_answer(text):
    """Extract A-E from model output — matches 04_eval_vllm.py."""
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    if text.startswith('</think>'):
        text = text[len('</think>'):].strip()
    text_upper = text.upper()

    m = re.search(r'정답[은:]?\s*\**\s*([A-Ea-e])', text)
    if m:
        return m.group(1).upper()
    m = re.search(r'[Aa]nswer[:\s]+\**\s*([A-Ea-e])', text)
    if m:
        return m.group(1).upper()
    first_line = text_upper.split('\n')[0].strip()
    if first_line and first_line[0] in "ABCDE":
        return first_line[0]
    m = re.search(r'[(\s]([A-E])[)\s.,]', text_upper)
    if m:
        return m.group(1)
    for ch in first_line:
        if ch in "ABCDE":
            return ch
    all_matches = re.findall(r'(?:정답|답)[^A-E]*([A-E])', text_upper)
    if all_matches:
        return all_matches[-1]
    for ch in text_upper:
        if ch in "ABCDE":
            return ch
    return ""


def eval_mcqa(base_url, hf_id, max_workers=16):
    print(f"  Loading dataset: {hf_id}")
    samples = load_mcqa(hf_id)
    if not samples:
        return {"type": "mcqa", "score": None, "error": "Failed to load"}

    messages_list = []
    for s in samples:
        content = format_mcqa_content(s)
        messages_list.append([{"role": "user", "content": content}])

    print(f"  Samples: {len(samples)}, sending with {max_workers} workers...")
    t0 = time.perf_counter()
    results = batch_chat(base_url, messages_list, max_tokens=512, max_workers=max_workers)
    gen_time = time.perf_counter() - t0

    correct = 0
    gen_tokens = 0
    for i, (text, usage) in enumerate(results):
        pred = parse_mcqa_answer(text)
        gen_tokens += usage.get("completion_tokens", 0)
        if pred == samples[i]["answer"]:
            correct += 1
        if i < 3:
            s = "O" if pred == samples[i]["answer"] else "X"
            print(f"    [{s}] pred={pred}, ref={samples[i]['answer']}, raw='{text[:80]}'")

    accuracy = correct / len(samples) if samples else 0
    print(f"  Accuracy: {accuracy:.4f} ({correct}/{len(samples)}), Time: {gen_time:.1f}s, Tokens: {gen_tokens}")
    return {
        "type": "mcqa", "score": round(accuracy, 4), "error": None,
        "num_samples": len(samples), "gen_time_s": round(gen_time, 4),
        "gen_tokens": gen_tokens,
    }


# ════════════════════════════════════════════════════════════
#  QA — with token-level context truncation
# ════════════════════════════════════════════════════════════

def load_qa(hf_id):
    ds = load_dataset_flex(hf_id)
    cols = ds.column_names
    has_prompt = "prompt" in cols

    ctx_col = _detect_col(ds, ["context", "passage", "document", "text", "input"])
    q_col = _detect_col(ds, ["question", "query", "problem"])
    a_col = _detect_col(ds, ["answer", "answers", "response", "output", "gold", "reference"])

    print(f"  Columns: {cols}")
    print(f"  Detected: ctx={ctx_col}, q={q_col}, a={a_col}, has_prompt={has_prompt}")

    samples = []
    for row in ds:
        context = str(row.get(ctx_col, "")) if ctx_col else ""
        question = str(row.get(q_col, "")) if q_col else ""
        answer = row.get(a_col, "") if a_col else ""
        if isinstance(answer, list):
            answer = answer[0] if answer else ""
        answer = str(answer)
        raw_prompt = str(row.get("prompt", "")) if has_prompt else ""
        titles_raw = row.get("titles", "")
        if isinstance(titles_raw, list):
            titles_str = ", ".join(str(t) for t in titles_raw)
        else:
            titles_str = str(titles_raw)
        samples.append({
            "context": context, "question": question,
            "answer": answer, "raw_prompt": raw_prompt, "titles": titles_str,
        })
    return samples


def _build_qa_content(sample, context_override=None):
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


def format_qa_content(sample, tokenizer, max_prompt_tokens=3500):
    """Format QA user content with token-level truncation."""
    content = _build_qa_content(sample)

    # Check token count (approximate: chat template adds ~20 tokens overhead)
    token_ids = tokenizer.encode(content)
    if len(token_ids) > max_prompt_tokens:
        # Binary search for max context length that fits
        context = sample["context"]
        lo, hi = 0, len(context)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            test_content = _build_qa_content(sample, context_override=context[:mid])
            if len(tokenizer.encode(test_content)) <= max_prompt_tokens:
                lo = mid
            else:
                hi = mid - 1
        content = _build_qa_content(sample, context_override=context[:lo])

    return content


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


def eval_qa(base_url, hf_id, tokenizer, max_model_len=16384, max_workers=8):
    print(f"  Loading dataset: {hf_id}")
    samples = load_qa(hf_id)
    if not samples:
        return {"type": "qa", "score": None, "error": "Failed to load"}

    max_prompt_tokens = max_model_len - 1024  # leave room for generation
    messages_list = []
    for s in samples:
        content = format_qa_content(s, tokenizer, max_prompt_tokens=max_prompt_tokens)
        messages_list.append([{"role": "user", "content": content}])

    print(f"  Samples: {len(samples)}, sending with {max_workers} workers...")
    t0 = time.perf_counter()
    # max_tokens=None → vLLM auto-fills with (max_model_len - prompt_tokens)
    results = batch_chat(base_url, messages_list, max_tokens=None, max_workers=max_workers)
    gen_time = time.perf_counter() - t0

    scores = []
    gen_tokens = 0
    for i, (text, usage) in enumerate(results):
        gen_tokens += usage.get("completion_tokens", 0)
        # Strip thinking tags, take first line
        text_clean = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        first_line = text_clean.split("\n")[0].strip()
        f1 = compute_f1(first_line, samples[i]["answer"])
        scores.append(f1)
        if i < 3:
            print(f"    F1={f1:.3f}, pred='{first_line[:60]}', ref='{samples[i]['answer'][:60]}'")

    avg_score = sum(scores) / len(scores) if scores else 0
    print(f"  F1: {avg_score:.4f}, Time: {gen_time:.1f}s, Tokens: {gen_tokens}")
    return {
        "type": "qa", "score": round(avg_score, 4), "error": None,
        "num_samples": len(scores), "gen_time_s": round(gen_time, 4),
        "gen_tokens": gen_tokens,
    }


# ════════════════════════════════════════════════════════════
#  Judge — handles KoMT-Bench 'turns' column
# ════════════════════════════════════════════════════════════

def eval_judge(base_url, hf_id, max_workers=8):
    print(f"  Loading dataset: {hf_id}")
    ds = load_dataset_flex(hf_id)

    cols = ds.column_names
    has_turns = "turns" in cols
    q_col = _detect_col(ds, ["question", "prompt", "input", "query", "instruction", "problem"])
    print(f"  Columns: {cols}, has_turns={has_turns}, q_col={q_col}")

    messages_list = []
    for row in ds:
        if has_turns:
            turns = row.get("turns", [])
            if isinstance(turns, list) and turns:
                question = str(turns[0])
            else:
                question = str(turns)
        elif q_col:
            question = str(row.get(q_col, ""))
        else:
            question = str(row)
        messages_list.append([{"role": "user", "content": question}])

    print(f"  Samples: {len(messages_list)}, sending with {max_workers} workers...")
    t0 = time.perf_counter()
    # max_tokens=None → vLLM auto-fills with (max_model_len - prompt_tokens)
    results = batch_chat(base_url, messages_list, max_tokens=None, max_workers=max_workers)
    gen_time = time.perf_counter() - t0

    gen_tokens = sum(u.get("completion_tokens", 0) for _, u in results)
    print(f"  Time: {gen_time:.1f}s, Tokens: {gen_tokens}")
    if results:
        print(f"    Output[0]: '{results[0][0][:80]}...'")
    return {
        "type": "judge", "score": None, "error": None,
        "num_samples": len(messages_list), "gen_time_s": round(gen_time, 4),
        "gen_tokens": gen_tokens,
    }


# ════════════════════════════════════════════════════════════
#  Speed test
# ════════════════════════════════════════════════════════════

def run_speed_test(base_url, n_runs=3):
    """Sequential speed test — single request latency."""
    speed_prompts = [
        "대한민국의 수도는 어디인가요?",
        "What is machine learning?",
        "Python에서 리스트를 정렬하는 방법을 설명해주세요.",
        "인공지능의 장점과 단점을 설명해주세요.",
        "Write a Python function to calculate fibonacci numbers.",
        "지구온난화의 원인과 해결방안은 무엇인가요?",
        "Explain quantum computing in simple terms.",
        "한국의 전통 음식에 대해 소개해주세요.",
    ]

    print("  Warmup...")
    for p in speed_prompts[:2]:
        chat_completion(base_url, [{"role": "user", "content": p}], max_tokens=256)

    all_times, all_tokens = [], []
    for run_i in range(n_runs):
        total_tokens = 0
        t0 = time.perf_counter()
        for p in speed_prompts:
            _, usage = chat_completion(base_url, [{"role": "user", "content": p}], max_tokens=256)
            total_tokens += usage.get("completion_tokens", 0)
        t1 = time.perf_counter()
        all_times.append(t1 - t0)
        all_tokens.append(total_tokens)
        print(f"  Run {run_i+1}: {total_tokens} tokens in {t1-t0:.3f}s = {total_tokens/(t1-t0):.0f} tok/s")

    median_idx = sorted(range(len(all_times)), key=lambda i: all_times[i])[len(all_times)//2]
    return {
        "total_time_s": round(all_times[median_idx], 4),
        "total_tokens": all_tokens[median_idx],
        "sec_per_token": round(all_times[median_idx] / all_tokens[median_idx], 6),
        "tokens_per_sec": round(all_tokens[median_idx] / all_times[median_idx], 2),
        "all_times": [round(t, 4) for t in all_times],
        "all_tokens": all_tokens,
    }


# ════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--lane-id", type=str, default=None)
    parser.add_argument("--port", type=int, default=8199)
    parser.add_argument("--max-model-len", type=int, default=None,
                        help="Override max_model_len (default: from model config)")
    parser.add_argument("--skip-bench", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    model_path = args.model if os.path.isabs(args.model) else os.path.join(ROOT, args.model)
    lane_id = args.lane_id or os.path.basename(model_path)
    base_url = f"http://localhost:{args.port}"

    out_dir = os.path.join(ROOT, "results", f"{lane_id}_srv")
    out_path = os.path.join(out_dir, "metrics.json")
    if os.path.exists(out_path) and not args.force:
        print(f"[SKIP] {out_path} exists. Use --force.")
        return

    with open(os.path.join(model_path, "config.json")) as f:
        model_cfg = json.load(f)

    if args.max_model_len:
        max_model_len = args.max_model_len
    else:
        max_model_len = model_cfg.get("max_position_embeddings", 16384)
    # Cap at 16384 to match server behavior
    max_model_len = min(max_model_len, 16384)

    # Load tokenizer for QA truncation
    from transformers import AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    except:
        base_path = os.path.normpath(os.path.join(ROOT, "../base_model"))
        tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)

    print(f"\n{'='*60}")
    print(f"  Server-mode eval: {model_path}")
    print(f"  Lane: {lane_id}_srv")
    print(f"  max_model_len: {max_model_len}")
    print(f"  Workers: {args.workers}")
    print(f"{'='*60}")

    # Start vLLM server
    print("\n[SERVER] Starting vLLM API server...")
    server_cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--served-model-name", "model",
        "--tensor-parallel-size", "1",
        "--gpu-memory-utilization", "0.85",
        "--max-model-len", str(max_model_len),
        "--trust-remote-code",
        "--port", str(args.port),
        "--disable-log-requests",
    ]
    server_log = open("/tmp/vllm_server.log", "w")
    server_proc = subprocess.Popen(server_cmd, stdout=server_log, stderr=subprocess.STDOUT)

    try:
        print("[SERVER] Waiting for ready...")
        if not wait_for_server(base_url, timeout=300):
            print("[SERVER] FAILED to start!")
            # Print last few lines of server log
            try:
                with open("/tmp/vllm_server.log") as f:
                    lines = f.readlines()
                    for line in lines[-20:]:
                        print(f"  [LOG] {line.rstrip()}")
            except:
                pass
            server_proc.kill()
            return
        print("[SERVER] Ready!\n")

        # Speed test (sequential)
        print("[SPEED] Running speed benchmark (sequential)...")
        speed = run_speed_test(base_url, n_runs=3)
        print(f"[SPEED] Median: {speed['total_tokens']} tokens in {speed['total_time_s']:.3f}s = {speed['tokens_per_sec']:.0f} tok/s\n")

        # Benchmarks (concurrent)
        benchmarks = {}
        total_gen_time = 0
        total_gen_tokens = 0

        if not args.skip_bench:
            cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", "lanes.yaml")))
            benchmarks_cfg = cfg["datasets"]["benchmarks"]

            for bench_name, bench_cfg in benchmarks_cfg.items():
                print(f"\n[BENCH] {bench_name} ({bench_cfg['type']})")
                try:
                    if bench_cfg["type"] == "mcqa":
                        r = eval_mcqa(base_url, bench_cfg["hf_id"], max_workers=args.workers)
                    elif bench_cfg["type"] == "qa":
                        r = eval_qa(base_url, bench_cfg["hf_id"], tokenizer,
                                    max_model_len=max_model_len, max_workers=args.workers)
                    elif bench_cfg["type"] == "judge":
                        r = eval_judge(base_url, bench_cfg["hf_id"], max_workers=args.workers)
                    else:
                        continue
                    benchmarks[bench_name] = r
                    total_gen_time += r.get("gen_time_s", 0)
                    total_gen_tokens += r.get("gen_tokens", 0)
                except Exception as e:
                    print(f"  ERROR: {e}")
                    import traceback; traceback.print_exc()
                    benchmarks[bench_name] = {"type": bench_cfg["type"], "score": None, "error": str(e)}

        # Aggregate
        available_scores = []
        weights = []
        if not args.skip_bench:
            benchmarks_cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", "lanes.yaml")))["datasets"]["benchmarks"]
            for bn, br in benchmarks.items():
                if br.get("score") is not None:
                    available_scores.append(br["score"])
                    weights.append(benchmarks_cfg[bn].get("weight", 1.0))

        if available_scores and sum(weights) > 0:
            perf = sum(s * w for s, w in zip(available_scores, weights)) / sum(weights)
        else:
            perf = None

        speed["bench_gen_time_s"] = round(total_gen_time, 4)
        speed["bench_gen_tokens"] = total_gen_tokens
        if total_gen_tokens > 0:
            speed["bench_sec_per_token"] = round(total_gen_time / total_gen_tokens, 6)

        results = {
            "model_path": model_path,
            "lane_id": f"{lane_id}_srv",
            "status": "OK",
            "benchmarks": benchmarks,
            "speed": speed,
            "perf_aggregate": round(perf, 4) if perf else None,
        }

        os.makedirs(out_dir, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        # Summary
        spt_base = 0.000552
        spt = speed["sec_per_token"]
        perf_norm = perf / 0.325 if perf else 0
        speed_norm = 1 - (spt / spt_base)
        score = 0.5 * perf_norm + 0.5 * speed_norm

        print(f"\n{'='*60}")
        print(f"  SUMMARY: {lane_id}_srv")
        print(f"  perf_aggregate: {results['perf_aggregate']}")
        for bn, br in benchmarks.items():
            print(f"    {bn}: {br.get('score')}")
        print(f"  Sequential tok/s: {speed['tokens_per_sec']}")
        if total_gen_tokens > 0:
            bench_tps = total_gen_tokens / total_gen_time
            print(f"  Bench throughput: {bench_tps:.0f} tok/s (concurrent)")
        print(f"  PerfNorm: {perf_norm:.4f}")
        print(f"  SpeedNorm (seq): {speed_norm:.4f}")
        print(f"  Estimated Score (seq): {score:.4f}")
        print(f"  Results: {out_path}")
        print(f"{'='*60}")

    finally:
        print("\n[SERVER] Shutting down...")
        server_proc.send_signal(signal.SIGTERM)
        try:
            server_proc.wait(timeout=30)
        except:
            server_proc.kill()
        server_log.close()
        print("[SERVER] Done.")


if __name__ == "__main__":
    main()
