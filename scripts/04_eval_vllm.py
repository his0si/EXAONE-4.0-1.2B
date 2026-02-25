#!/usr/bin/env python3
"""
04_eval_vllm.py — vLLM-based evaluation on benchmarks + speed measurement.

Mirrors competition server conditions exactly:
  - apply_chat_template = true (no system prompt, no few-shot, no enable_thinking)
  - temperature = 0.0
  - max_gen_toks = 16384
  - tensor_parallel_size = 1, gpu_memory_utilization = 0.85

Usage:
  python scripts/04_eval_vllm.py --model models/lane01_gptq_w4a16 --lane-id lane01_gptq_w4a16
  python scripts/04_eval_vllm.py --model ../base_model --baseline
"""
import os, sys, re, json, time, argparse, warnings
import yaml
from collections import defaultdict

warnings.filterwarnings("ignore", message=".*incorrect regex.*")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load_config(path="configs/lanes.yaml"):
    with open(os.path.join(ROOT, path)) as f:
        return yaml.safe_load(f)


# ════════════════════════════════════════════════════════════
#  Benchmark Loaders & Scorers
# ════════════════════════════════════════════════════════════

def _detect_columns(ds, candidates):
    """Find the first matching column name from candidates."""
    cols = ds.column_names
    for c in candidates:
        if c in cols:
            return c
    return None


def _load_dataset_flexible(hf_id):
    """Try various strategies to load a HF dataset."""
    from datasets import load_dataset

    # Strategy 1: direct split loading
    for split in ["test", "validation", "train"]:
        try:
            ds = load_dataset(hf_id, split=split)
            return ds, split
        except:
            continue

    # Strategy 2: load all and pick first split
    try:
        info = load_dataset(hf_id)
        available = list(info.keys())
        if available:
            return info[available[0]], available[0]
    except:
        pass

    # Strategy 3: try common subset names (some datasets have configs)
    for config_name in [None, "default", "ko", "korean"]:
        for split in ["test", "validation", "train"]:
            try:
                ds = load_dataset(hf_id, config_name, split=split)
                return ds, split
            except:
                continue

    raise RuntimeError(f"Could not load dataset: {hf_id}")


def load_mcqa_benchmark(hf_id, max_samples=None):
    """Load multiple-choice QA benchmark. Returns list of dicts with
    'question', 'choices' (list), 'answer' (letter A-E)."""
    ds, split = _load_dataset_flexible(hf_id)

    if max_samples and len(ds) > max_samples:
        ds = ds.select(range(max_samples))

    # Detect columns
    q_col = _detect_columns(ds, ["question", "query", "input", "prompt", "problem"])
    c_col = _detect_columns(ds, ["choices", "options", "candidates", "choice"])
    a_col = _detect_columns(ds, ["answer", "label", "correct", "answer_key", "gold",
                                  "solution"])  # ← solution 추가 (KMMLU 계열)

    if not q_col:
        print(f"  [MCQA] Cannot detect question column. Cols: {ds.column_names}")
        return None

    print(f"  [MCQA] Columns detected: q={q_col}, c={c_col}, a={a_col}")

    samples = []
    for row in ds:
        question = str(row[q_col]) if q_col else ""
        choices = row.get(c_col, []) if c_col else []
        answer_raw = row.get(a_col, "") if a_col else ""

        # Normalize choices to list of strings
        if isinstance(choices, dict):
            choices = list(choices.values())
        elif isinstance(choices, str):
            choices = [c.strip() for c in choices.split("\n") if c.strip()]

        # Normalize answer — handle various formats
        answer = ""
        if isinstance(answer_raw, int):
            # 0-indexed integer → letter
            answer = chr(ord("A") + answer_raw)
        elif isinstance(answer_raw, str):
            s = answer_raw.strip()
            if s.isdigit():
                # KMMLU-Pro/Redux: solution is 1-indexed string ("1"~"5")
                idx = int(s)
                if idx >= 1 and choices and idx <= len(choices):
                    # 1-indexed → letter (1→A, 2→B, ...)
                    answer = chr(ord("A") + idx - 1)
                elif idx == 0:
                    answer = "A"  # 0-indexed
                else:
                    answer = chr(ord("A") + min(idx, len(choices) - 1))
            elif s and s[0].upper() in "ABCDE":
                answer = s[0].upper()
            else:
                answer = s.upper()

        samples.append({"question": question, "choices": choices, "answer": answer})

    # Sanity check: show answer distribution
    from collections import Counter
    ans_dist = Counter(s["answer"] for s in samples)
    print(f"  [MCQA] Answer distribution: {dict(ans_dist)}")

    return samples


def format_mcqa_prompt(sample, tokenizer):
    """Format MCQA as chat — matches competition (no system prompt, no few-shot).
    최소한의 프롬프트로 모델의 원래 성능을 최대한 발휘."""
    labels = "ABCDE"
    choice_text = "\n".join(
        f"{labels[i]}. {c}" for i, c in enumerate(sample["choices"])
    ) if sample["choices"] else ""

    content = (
        f"{sample['question']}\n"
        f"{choice_text}\n"
        f"정답:"
    )
    messages = [{"role": "user", "content": content}]
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False)


def strip_thinking(text):
    """Remove <think>...</think> block from model output (safety fallback)."""
    stripped = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    if stripped.startswith('</think>'):
        stripped = stripped[len('</think>'):].strip()
    return stripped


def parse_mcqa_answer(text):
    """Extract A-E from model output — 다양한 답변 형식 처리."""
    text = strip_thinking(text).strip()
    text_upper = text.upper()

    # 1) "정답: X" 또는 "정답은 X" 패턴 (가장 명확)
    m = re.search(r'정답[은:]?\s*\**\s*([A-Ea-e])', text)
    if m:
        return m.group(1).upper()

    # 2) "Answer: X" 패턴
    m = re.search(r'[Aa]nswer[:\s]+\**\s*([A-Ea-e])', text)
    if m:
        return m.group(1).upper()

    # 3) 첫 글자가 A-E
    first_line = text_upper.split('\n')[0].strip()
    if first_line and first_line[0] in "ABCDE":
        return first_line[0]

    # 4) "(A)" or "A)" or "A." 패턴
    m = re.search(r'[(\s]([A-E])[)\s.,]', text_upper)
    if m:
        return m.group(1)

    # 5) 첫 줄에서 A-E 찾기
    for ch in first_line:
        if ch in "ABCDE":
            return ch

    # 6) 전체 텍스트에서 마지막 A-E (모델이 설명 후 답변하는 경우)
    all_matches = re.findall(r'(?:정답|답)[^A-E]*([A-E])', text_upper)
    if all_matches:
        return all_matches[-1]

    # 7) 최후 수단: 텍스트에서 첫 A-E
    for ch in text_upper:
        if ch in "ABCDE":
            return ch
    return ""


def score_mcqa(predictions, references):
    """Accuracy for MCQA."""
    if not predictions:
        return 0.0
    correct = sum(1 for p, r in zip(predictions, references)
                  if p.strip().upper() == r.strip().upper())
    return correct / len(predictions)


def load_qa_benchmark(hf_id, max_context_len=3072, max_samples=None):
    """Load QA benchmark. Returns list of dicts with
    'context', 'question', 'answer'.
    Ko-LongRAG has a 'prompt' column that is a pre-formatted template."""
    ds, split = _load_dataset_flexible(hf_id)

    if max_samples and len(ds) > max_samples:
        ds = ds.select(range(max_samples))

    cols = ds.column_names
    has_prompt_col = "prompt" in cols

    ctx_col = _detect_columns(ds, ["context", "passage", "document", "text", "input"])
    q_col = _detect_columns(ds, ["question", "query", "problem"])
    a_col = _detect_columns(ds, ["answer", "answers", "response", "output", "gold",
                                  "reference"])

    print(f"  [QA] Columns: {cols}")
    print(f"  [QA] Detected: ctx={ctx_col}, q={q_col}, a={a_col}, has_prompt={has_prompt_col}")

    samples = []
    for row in ds:
        context = str(row.get(ctx_col, "")) if ctx_col else ""
        question = str(row.get(q_col, "")) if q_col else ""
        answer = row.get(a_col, "") if a_col else ""
        if isinstance(answer, list):
            answer = answer[0] if answer else ""
        answer = str(answer)

        raw_prompt = str(row.get("prompt", "")) if has_prompt_col else ""

        # titles: Ko-LongRAG에서 리스트 → 쉼표 구분 문자열로 변환
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


def _build_qa_content(sample, context_override=None):
    """Build QA prompt content string."""
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


def format_qa_prompt(sample, tokenizer, max_prompt_tokens=3500):
    """Format QA prompt — matches competition server (plain chat, no extras).
    Applies token-level truncation to fit max_model_len."""
    content = _build_qa_content(sample)
    messages = [{"role": "user", "content": content}]
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False)

    # Token-level truncation: if prompt exceeds limit, truncate context
    token_ids = tokenizer.encode(prompt)
    if len(token_ids) > max_prompt_tokens:
        context = sample["context"]
        lo, hi = 0, len(context)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            test_content = _build_qa_content(sample, context_override=context[:mid])
            test_msgs = [{"role": "user", "content": test_content}]
            test_prompt = tokenizer.apply_chat_template(
                test_msgs, add_generation_prompt=True, tokenize=False)
            if len(tokenizer.encode(test_prompt)) <= max_prompt_tokens:
                lo = mid
            else:
                hi = mid - 1

        content = _build_qa_content(sample, context_override=context[:lo])
        messages = [{"role": "user", "content": content}]
        prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False)

    return prompt


def compute_f1(prediction, reference):
    """Token-level F1 (Korean-aware, handles both word and character level)."""
    # 먼저 앞뒤 공백 및 불필요한 구두점 제거
    pred_clean = re.sub(r'[^\w\s가-힣]', ' ', prediction.lower()).strip()
    ref_clean = re.sub(r'[^\w\s가-힣]', ' ', reference.lower()).strip()

    pred_tokens = pred_clean.split()
    ref_tokens = ref_clean.split()
    if not pred_tokens or not ref_tokens:
        return 0.0

    # 단어 단위 F1
    common = set(pred_tokens) & set(ref_tokens)
    if not common:
        # 음절 단위로 fallback (한국어 짧은 답변)
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


def score_qa(predictions, references):
    if not predictions:
        return 0.0
    scores = [compute_f1(p, r) for p, r in zip(predictions, references)]
    return sum(scores) / len(scores)


def load_judge_benchmark(hf_id, max_samples=None):
    """Load MT-Bench style benchmark (generation-only).
    KoMT-Bench has 'turns' column with multi-turn prompts."""
    ds, split = _load_dataset_flexible(hf_id)

    if max_samples and len(ds) > max_samples:
        ds = ds.select(range(max_samples))

    cols = ds.column_names
    print(f"  [Judge] Columns: {cols}, split={split}")

    # KoMT-Bench: turns 컬럼 (리스트 of 문자열)
    has_turns = "turns" in cols
    q_col = _detect_columns(ds, ["question", "prompt", "input", "query",
                                  "instruction", "problem"])

    samples = []
    for row in ds:
        if has_turns:
            turns = row.get("turns", [])
            if isinstance(turns, list) and turns:
                # 첫 번째 턴만 사용 (단일 턴 평가)
                question = str(turns[0])
            else:
                question = str(turns)
        elif q_col:
            question = str(row.get(q_col, ""))
        else:
            question = str(row)

        ref = row.get("reference", None)
        if isinstance(ref, list):
            ref = ref[0] if ref else None
        samples.append({"question": question, "reference": ref})
    return samples


def format_judge_prompt(sample, tokenizer):
    """Format judge prompt — plain chat, no system prompt."""
    messages = [{"role": "user", "content": sample["question"]}]
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False)


# ════════════════════════════════════════════════════════════
#  Gemini Judge for KoMT-Bench
# ════════════════════════════════════════════════════════════

def judge_with_gemini(question, answer, reference, api_key, model="gemini-2.0-flash"):
    """Use Gemini API to judge a single KoMT-Bench response. Returns score 1-10."""
    import requests

    if reference:
        judge_prompt = f"""[System]
Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider correctness and helpfulness. You will be given a reference answer and the assistant's answer. Begin your evaluation by comparing the assistant's answer with the reference answer. Identify and correct any mistakes. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".

[Question]
{question}

[The Start of Reference Answer]
{reference}
[The End of Reference Answer]

[The Start of Assistant's Answer]
{answer}
[The End of Assistant's Answer]"""
    else:
        judge_prompt = f"""[System]
Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".

[Question]
{question}

[The Start of Assistant's Answer]
{answer}
[The End of Assistant's Answer]"""

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": judge_prompt}]}],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 1024}
    }

    try:
        resp = requests.post(url, json=payload, timeout=60)
        resp.raise_for_status()
        text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        # Parse [[rating]]
        match = re.search(r'\[\[(\d+(?:\.\d+)?)\]\]', text)
        if match:
            return float(match.group(1))
        # Fallback: "Rating: X"
        match = re.search(r'[Rr]ating:\s*(\d+(?:\.\d+)?)', text)
        if match:
            return float(match.group(1))
        # Fallback: any standalone number at end
        match = re.search(r'(\d+(?:\.\d+)?)\s*$', text.strip())
        if match:
            return float(match.group(1))
        print(f"    [Judge] Could not parse score from: {text[:100]}")
        return None
    except Exception as e:
        print(f"    [Judge ERROR] {e}")
        return None


def run_gemini_judge(samples, outputs, api_key, model="gemini-2.0-flash"):
    """Judge all KoMT-Bench responses with Gemini. Returns average score (0-1 scale)."""
    import time as _time
    scores = []
    for i, (sample, output) in enumerate(zip(samples, outputs)):
        answer = output.outputs[0].text.strip()
        score = judge_with_gemini(
            sample["question"], answer, sample.get("reference"), api_key, model)
        if score is not None:
            scores.append(score)
            if i < 5 or (i + 1) % 20 == 0:
                print(f"    [{i+1}/{len(samples)}] score={score:.1f} | q='{sample['question'][:50]}'")
        else:
            print(f"    [{i+1}/{len(samples)}] score=FAILED | q='{sample['question'][:50]}'")
        # Rate limit: ~30 RPM for free tier
        if (i + 1) % 10 == 0:
            _time.sleep(1)
    if scores:
        avg = sum(scores) / len(scores)
        print(f"  [Judge] Gemini scores: {len(scores)}/{len(samples)} rated, avg={avg:.2f}/10 → {avg/10:.4f}")
        return round(avg / 10, 4)  # Normalize to 0-1
    return None


# ════════════════════════════════════════════════════════════
#  vLLM Evaluation Engine
# ════════════════════════════════════════════════════════════

def run_vllm_eval(model_path, cfg, lane_id=None, is_baseline=False, gemini_key=None):
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    vcfg = cfg["vllm"]
    benchmarks_cfg = cfg["datasets"]["benchmarks"]
    base_path = os.path.normpath(os.path.join(ROOT, cfg["project"]["base_model"]))

    # 토크나이저 로드: 모델 디렉토리 우선, 없으면 base model에서
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, local_files_only=True)
    except Exception:
        print(f"[EVAL] 모델 토크나이저 로드 실패, base model에서 로드")
        tokenizer = AutoTokenizer.from_pretrained(
            base_path, trust_remote_code=True)

    print(f"\n{'='*60}")
    print(f"  Evaluating: {model_path}")
    print(f"  Lane: {lane_id or 'baseline'}")
    print(f"  (Competition-matched: no system prompt, no few-shot, no thinking)")
    print(f"{'='*60}")

    # 대회 서버 설정 그대로 적용
    print("[EVAL] Loading model with vLLM...")
    try:
        llm = LLM(
            model=model_path,
            trust_remote_code=True,
            tensor_parallel_size=vcfg["tensor_parallel_size"],
            gpu_memory_utilization=vcfg["gpu_memory_utilization"],
            max_model_len=vcfg["max_model_len"],
        )
    except Exception as e:
        print(f"[EVAL] vLLM load FAILED: {e}")
        return {"status": "FAILED", "error": str(e)}

    # 대회 서버: temperature 고정, max_gen_toks = 16384
    sampling = SamplingParams(
        temperature=vcfg["sampling"]["temperature"],
        max_tokens=vcfg["sampling"]["max_tokens"],
    )
    # MCQA: 모델이 설명 후 답변하는 경우 대비하여 충분한 토큰 확보
    sampling_mcqa = SamplingParams(temperature=0.0, max_tokens=512)

    results = {
        "model_path": model_path,
        "lane_id": lane_id,
        "is_baseline": is_baseline,
        "status": "OK",
        "benchmarks": {},
        "speed": {},
    }

    total_gen_time = 0.0
    total_gen_tokens = 0

    # ── Run each benchmark ──
    for bench_name, bench_cfg in benchmarks_cfg.items():
        print(f"\n[BENCH] {bench_name} ({bench_cfg['type']})")
        bench_result = {"type": bench_cfg["type"], "score": None, "error": None,
                        "num_samples": 0, "gen_time_s": 0, "gen_tokens": 0}

        try:
            if bench_cfg["type"] == "mcqa":
                samples = load_mcqa_benchmark(
                    bench_cfg["hf_id"], max_samples=None)  # 전체 사용
                if not samples:
                    raise RuntimeError("Failed to load MCQA data")

                prompts = [format_mcqa_prompt(s, tokenizer) for s in samples]
                bench_result["num_samples"] = len(prompts)

                t0 = time.perf_counter()
                outputs = llm.generate(prompts, sampling_mcqa)
                gen_time = time.perf_counter() - t0

                predictions = []
                gen_tokens = 0
                raw_outputs = []
                for o in outputs:
                    text = o.outputs[0].text
                    gen_tokens += len(o.outputs[0].token_ids)
                    parsed = parse_mcqa_answer(text)
                    predictions.append(parsed)
                    raw_outputs.append(text)

                references = [s["answer"] for s in samples]
                accuracy = score_mcqa(predictions, references)

                bench_result["score"] = round(accuracy, 4)
                bench_result["gen_time_s"] = round(gen_time, 4)
                bench_result["gen_tokens"] = gen_tokens
                total_gen_time += gen_time
                total_gen_tokens += gen_tokens

                correct = sum(1 for p, r in zip(predictions, references) if p == r)
                print(f"  Accuracy: {accuracy:.4f} ({correct}/{len(predictions)})")
                print(f"  Time: {gen_time:.2f}s, Tokens: {gen_tokens}")
                # 디버깅: 처음 5개 예시 출력
                for i in range(min(5, len(predictions))):
                    status = "✓" if predictions[i] == references[i] else "✗"
                    print(f"    [{status}] pred={predictions[i]}, ref={references[i]}, raw='{raw_outputs[i][:60]}'")


            elif bench_cfg["type"] == "qa":
                max_ctx = bench_cfg.get("max_context_len", 3072)
                samples = load_qa_benchmark(
                    bench_cfg["hf_id"], max_context_len=max_ctx, max_samples=None)  # 전체 사용
                if not samples:
                    raise RuntimeError("Failed to load QA data")

                # max_prompt_tokens = max_model_len - gen buffer(1024)
                max_prompt_toks = vcfg["max_model_len"] - 1024
                prompts = [format_qa_prompt(s, tokenizer,
                           max_prompt_tokens=max_prompt_toks) for s in samples]
                bench_result["num_samples"] = len(prompts)

                t0 = time.perf_counter()
                outputs = llm.generate(prompts, sampling)
                gen_time = time.perf_counter() - t0

                predictions = []
                gen_tokens = 0
                for o in outputs:
                    text = strip_thinking(o.outputs[0].text).strip()
                    # 첫 줄만 추출 (간결한 답변 기대)
                    first_line = text.split("\n")[0].strip()
                    gen_tokens += len(o.outputs[0].token_ids)
                    predictions.append(first_line)

                references = [s["answer"] for s in samples]
                f1 = score_qa(predictions, references)

                bench_result["score"] = round(f1, 4)
                bench_result["gen_time_s"] = round(gen_time, 4)
                bench_result["gen_tokens"] = gen_tokens
                total_gen_time += gen_time
                total_gen_tokens += gen_tokens

                print(f"  F1: {f1:.4f}")
                print(f"  Time: {gen_time:.2f}s, Tokens: {gen_tokens}")
                # 디버깅: 처음 5개 예시 출력
                for i in range(min(5, len(predictions))):
                    individual_f1 = compute_f1(predictions[i], references[i])
                    print(f"    [F1={individual_f1:.3f}] pred='{predictions[i][:80]}' | ref='{references[i][:80]}'")

            elif bench_cfg["type"] == "judge":
                fallback = bench_cfg.get("fallback", "skip_perf")
                samples = load_judge_benchmark(bench_cfg["hf_id"], max_samples=None)  # 전체 사용
                if not samples:
                    raise RuntimeError("Failed to load judge data")

                prompts = [format_judge_prompt(s, tokenizer) for s in samples]
                bench_result["num_samples"] = len(prompts)

                t0 = time.perf_counter()
                outputs = llm.generate(prompts, sampling)
                gen_time = time.perf_counter() - t0

                gen_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
                bench_result["gen_time_s"] = round(gen_time, 4)
                bench_result["gen_tokens"] = gen_tokens
                total_gen_time += gen_time
                total_gen_tokens += gen_tokens

                print(f"  Time: {gen_time:.2f}s, Tokens: {gen_tokens}")

                # Judge with Gemini if API key provided
                if gemini_key:
                    print(f"  [Judge] Scoring with Gemini...")
                    judge_score = run_gemini_judge(samples, outputs, gemini_key)
                    bench_result["score"] = judge_score
                    if judge_score is not None:
                        print(f"  Score: {judge_score:.4f} (Gemini judge)")
                    else:
                        print(f"  Score: FAILED (Gemini judge returned no scores)")
                elif fallback == "skip_perf":
                    bench_result["score"] = None
                    print(f"  Score: SKIPPED (no judge model, use --gemini-key)")
                else:
                    bench_result["score"] = None

        except Exception as e:
            import traceback
            traceback.print_exc()
            bench_result["error"] = str(e)
            print(f"  ERROR: {e}")

        results["benchmarks"][bench_name] = bench_result

    # ── Speed measurement (dedicated run) ──
    print(f"\n[SPEED] Running speed benchmark ({vcfg['speed_runs']} runs)...")
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
    formatted_speed = []
    for p in speed_prompts:
        messages = [{"role": "user", "content": p}]
        formatted_speed.append(tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False))

    speed_sampling = SamplingParams(temperature=0.0, max_tokens=256)

    # Warmup
    _ = llm.generate(formatted_speed[:2], speed_sampling)

    all_times, all_tokens = [], []
    for run_i in range(vcfg["speed_runs"]):
        t0 = time.perf_counter()
        outs = llm.generate(formatted_speed, speed_sampling)
        t1 = time.perf_counter()
        tokens = sum(len(o.outputs[0].token_ids) for o in outs)
        all_times.append(t1 - t0)
        all_tokens.append(tokens)

    # Median
    median_idx = sorted(range(len(all_times)), key=lambda i: all_times[i])[len(all_times)//2]
    med_time = all_times[median_idx]
    med_tokens = all_tokens[median_idx]

    results["speed"] = {
        "total_time_s": round(med_time, 4),
        "total_tokens": med_tokens,
        "sec_per_token": round(med_time / med_tokens, 6) if med_tokens > 0 else 0,
        "tokens_per_sec": round(med_tokens / med_time, 2) if med_time > 0 else 0,
        "all_times": [round(t, 4) for t in all_times],
        "all_tokens": all_tokens,
    }

    # Aggregate benchmark gen time + tokens into speed section too
    results["speed"]["bench_gen_time_s"] = round(total_gen_time, 4)
    results["speed"]["bench_gen_tokens"] = total_gen_tokens
    if total_gen_tokens > 0:
        results["speed"]["bench_sec_per_token"] = round(
            total_gen_time / total_gen_tokens, 6)

    print(f"\n[SPEED] {med_tokens} tokens in {med_time:.3f}s "
          f"= {med_tokens/med_time:.1f} tok/s "
          f"({med_time/med_tokens*1000:.2f} ms/tok)")

    # Compute aggregate performance
    available_scores = []
    weights = []
    for bench_name, bench_cfg in benchmarks_cfg.items():
        br = results["benchmarks"].get(bench_name, {})
        if br.get("score") is not None:
            available_scores.append(br["score"])
            weights.append(bench_cfg.get("weight", 1.0))

    if available_scores:
        w_sum = sum(weights)
        perf = sum(s * w for s, w in zip(available_scores, weights)) / w_sum
        results["perf_aggregate"] = round(perf, 4)
    else:
        results["perf_aggregate"] = None

    # Clean up
    del llm
    import gc
    gc.collect()
    if hasattr(torch := __import__("torch"), "cuda"):
        torch.cuda.empty_cache()

    # Save
    if is_baseline:
        out_dir = os.path.join(ROOT, "results")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "baseline.json")
    else:
        out_dir = os.path.join(ROOT, "results", lane_id or "unknown")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "metrics.json")

    with open(out_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[EVAL] Results saved to {out_path}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to model directory")
    parser.add_argument("--lane-id", type=str, default=None)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--config", default="configs/lanes.yaml")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing results")
    parser.add_argument("--gemini-key", type=str, default=None,
                        help="Gemini API key for KoMT-Bench judge scoring")
    parser.add_argument("--server-mode", action="store_true",
                        help="Mimic server: use model's max_position_embeddings as max_model_len")
    args = parser.parse_args()

    cfg = load_config(os.path.join(ROOT, args.config))
    model_path = args.model if os.path.isabs(args.model) else os.path.join(ROOT, args.model)

    # --server-mode: read max_position_embeddings from model config
    if args.server_mode:
        model_config_path = os.path.join(model_path, "config.json")
        if os.path.exists(model_config_path):
            with open(model_config_path) as f:
                model_cfg = json.load(f)
            mpe = model_cfg.get("max_position_embeddings", 65536)
            cfg["vllm"]["max_model_len"] = mpe
            print(f"[SERVER-MODE] max_model_len = {mpe} (from model config)")

    if not os.path.isdir(model_path):
        print(f"ERROR: Model path not found: {model_path}")
        sys.exit(1)

    # 기존 결과 확인
    if not args.force:
        if args.baseline:
            existing = os.path.join(ROOT, "results", "baseline.json")
        else:
            existing = os.path.join(ROOT, "results", args.lane_id or "unknown", "metrics.json")
        if os.path.exists(existing):
            print(f"[EVAL] 결과 이미 존재: {existing} (use --force to re-run)")
            return

    run_vllm_eval(model_path, cfg, lane_id=args.lane_id, is_baseline=args.baseline,
                  gemini_key=args.gemini_key)


if __name__ == "__main__":
    main()
