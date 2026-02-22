"""Korean calibration data loader for GPTQ quantization."""
import random
from datasets import load_dataset, Dataset


KMMLU_CONFIGS = [
    'Accounting', 'Agricultural-Sciences', 'Aviation-Engineering-and-Maintenance',
    'Biology', 'Chemical-Engineering', 'Chemistry', 'Civil-Engineering',
    'Computer-Science', 'Construction', 'Criminal-Law', 'Ecology', 'Economics',
    'Education', 'Electrical-Engineering', 'Electronics-Engineering',
    'Energy-Management', 'Environmental-Science', 'Fashion', 'Food-Processing',
    'Gas-Technology-and-Engineering', 'Geomatics', 'Health', 'Industrial-Engineer',
    'Information-Technology', 'Interior-Architecture-and-Design', 'Law',
    'Machine-Design-and-Manufacturing', 'Management', 'Maritime-Engineering',
    'Marketing', 'Materials-Engineering', 'Mechanical-Engineering',
    'Nondestructive-Testing', 'Patent', 'Political-Science-and-Sociology',
    'Psychology', 'Public-Safety', 'Railway-and-Automotive-Engineering',
    'Real-Estate', 'Refrigerating-Machinery', 'Social-Welfare', 'Taxation',
    'Telecommunications-and-Wireless-Technology', 'Korean-History', 'Math',
]


def load_korean_calibration(tokenizer, num_samples=2048, include_manta=True, manta_ratio=0.5, seed=42):
    """
    Load Korean-benchmark calibration data for GPTQ.

    Args:
        tokenizer: EXAONE tokenizer
        num_samples: Total number of calibration samples
        include_manta: Whether to mix MANTA-1M data
        manta_ratio: Ratio of MANTA-1M data (0.5 = 50% KMMLU + 50% MANTA)
        seed: Random seed
    """
    random.seed(seed)
    answer_map = {1: 'A', 2: 'B', 3: 'C', 4: 'D'}

    if include_manta:
        kmmlu_count = int(num_samples * (1 - manta_ratio))
        manta_count = num_samples - kmmlu_count
    else:
        kmmlu_count = num_samples
        manta_count = 0

    # Load KMMLU data
    kmmlu_samples = []
    per_category = max(1, (kmmlu_count // len(KMMLU_CONFIGS)) + 1)

    print(f"[INFO] Loading KMMLU data ({kmmlu_count} samples from {len(KMMLU_CONFIGS)} categories)...")
    for config in KMMLU_CONFIGS:
        try:
            ds = load_dataset('HAERAE-HUB/KMMLU', config, split='train')
            items = list(ds.select(range(min(per_category, len(ds)))))
            for item in items:
                answer_letter = answer_map.get(item['answer'], 'A')
                answer_text = item.get(answer_letter, '')
                conversation = [
                    {"role": "user", "content": f"다음 문제의 정답을 선택하세요.\n\n{item['question']}\nA: {item['A']}\nB: {item['B']}\nC: {item['C']}\nD: {item['D']}"},
                    {"role": "assistant", "content": f"정답은 {answer_letter}번 '{answer_text}'입니다."}
                ]
                text = tokenizer.apply_chat_template(conversation, add_generation_prompt=False, tokenize=False)
                kmmlu_samples.append({"text": text})
        except Exception as e:
            print(f"  [WARN] Failed to load {config}: {e}")

    random.shuffle(kmmlu_samples)
    kmmlu_samples = kmmlu_samples[:kmmlu_count]
    print(f"  Loaded {len(kmmlu_samples)} KMMLU samples")

    # Load MANTA-1M data
    manta_samples = []
    if manta_count > 0:
        print(f"[INFO] Loading MANTA-1M data ({manta_count} samples)...")
        manta_ds = load_dataset('LGAI-EXAONE/MANTA-1M', split=f'train[:{manta_count}]')
        for item in manta_ds:
            text = tokenizer.apply_chat_template(
                item['conversations'],
                add_generation_prompt=True,
                tokenize=False,
            )
            manta_samples.append({"text": text})
        print(f"  Loaded {len(manta_samples)} MANTA samples")

    # Combine and shuffle
    all_samples = kmmlu_samples + manta_samples
    random.shuffle(all_samples)
    all_samples = all_samples[:num_samples]

    print(f"[INFO] Total calibration samples: {len(all_samples)}")
    return Dataset.from_list(all_samples)


def load_kmmlu_for_sft(tokenizer, num_samples=5000, seed=42):
    """
    Load KMMLU data formatted for SFT/LoRA training.
    Returns dataset with 'text' column ready for training.
    """
    random.seed(seed)
    answer_map = {1: 'A', 2: 'B', 3: 'C', 4: 'D'}

    samples = []
    per_category = max(1, (num_samples // len(KMMLU_CONFIGS)) + 1)

    print(f"[INFO] Loading KMMLU for SFT ({num_samples} samples)...")
    for config in KMMLU_CONFIGS:
        try:
            # Use train split for training data
            ds = load_dataset('HAERAE-HUB/KMMLU', config, split='train')
            items = list(ds.select(range(min(per_category, len(ds)))))
            for item in items:
                answer_letter = answer_map.get(item['answer'], 'A')
                answer_text = item.get(answer_letter, '')
                conversation = [
                    {"role": "user", "content": f"다음 문제의 정답을 선택하고 이유를 간단히 설명하세요.\n\n{item['question']}\nA: {item['A']}\nB: {item['B']}\nC: {item['C']}\nD: {item['D']}"},
                    {"role": "assistant", "content": f"정답: {answer_letter}\n\n{answer_letter}번 '{answer_text}'이(가) 정답입니다."}
                ]
                text = tokenizer.apply_chat_template(conversation, add_generation_prompt=False, tokenize=False)
                samples.append({"text": text})
        except Exception as e:
            print(f"  [WARN] Failed to load {config}: {e}")

    random.shuffle(samples)
    samples = samples[:num_samples]
    print(f"[INFO] Total SFT samples: {len(samples)}")
    return Dataset.from_list(samples)
