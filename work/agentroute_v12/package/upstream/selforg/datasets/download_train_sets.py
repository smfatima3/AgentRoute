# download_train_sets.py
# Usage:
#   python3 datasets/download_train_sets.py
#   python3 datasets/download_train_sets.py --only GSM8K MMLU MATH
#
# Mirrors Hugging Face dataset IDs/configs/splits used by datasets/build_test_dataset.py
# and writes training sets to datasets/data/<DATASET>_val.json for --require_val.

import argparse
import json
from pathlib import Path

from datasets import load_dataset

OUT_DIR = Path(__file__).resolve().parent / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# expand this dataset if needed
ALL_DATASETS = [
    "AIME-2024",
    "AQUA-RAT",
    "GPQA",
    "GSM-Hard",
    "GSM8K",
    "MATH",
    "MMLU",
    "MMLU-Pro",
]


def _try_load(candidates):
    last_err = None
    for ds_name, ds_cfg in candidates:
        for split in ("train", "validation"):
            try:
                if ds_cfg is None:
                    ds = load_dataset(ds_name, split=split, trust_remote_code=True)
                else:
                    ds = load_dataset(ds_name, ds_cfg, split=split, trust_remote_code=True)
                return ds, (ds_name, ds_cfg, split)
            except Exception as exc:
                last_err = exc
    raise RuntimeError(last_err)


def _load_for_dataset(dataset_name: str):
    if dataset_name == "MATH":
        return _try_load(
            [
                ("hendrycks/competition_math", None),
                ("EleutherAI/hendrycks_math", None),
                ("HuggingFaceH4/MATH-500", None),
            ]
        )
    if dataset_name == "GSM8K":
        return _try_load([("openai/gsm8k", "main")])
    if dataset_name == "AQUA-RAT":
        return _try_load([("deepmind/aqua_rat", "raw")])
    if dataset_name == "MMLU":
        return _try_load([("cais/mmlu", "all")])
    if dataset_name == "MMLU-Pro":
        return _try_load([("TIGER-Lab/MMLU-Pro", None)])
    if dataset_name == "GSM-Hard":
        return _try_load([("reasoning-machines/gsm-hard", None)])
    if dataset_name == "GPQA":
        return _try_load([("Idavidrein/gpqa", "gpqa_main")])
    if dataset_name == "AIME-2024":
        return _try_load([("Maxwell-Jia/AIME_2024", None)])
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def _format_aqua_query(example):
    query = example["question"]
    query += " Choose the correct answer from the following options:"
    for option in example["options"]:
        query += f"\n{option}"
    return query


def _format_mmlu_query(example):
    return (
        "The following is a multiple-choice question:\n"
        f"{example['question']}\n\n"
        "Choose the correct answer from the following options:\n"
        f"(A) {example['choices'][0]}\n"
        f"(B) {example['choices'][1]}\n"
        f"(C) {example['choices'][2]}\n"
        f"(D) {example['choices'][3]}"
    )


def _format_mmlu_gt(example):
    choice_list = ["A", "B", "C", "D"]
    return f"({choice_list[example['answer']]})"


def _format_mmlu_pro_query(example):
    option_list = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    query = "The following is a multiple-choice question:\n"
    query += example["question"]
    query += "\n\nChoose the correct answer from the following options:"
    for idx, option in enumerate(example["options"]):
        query += f"\n({option_list[idx]}) {option}"
    return query


def _format_mmlu_pro_gt(example):
    option_list = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    idx = example["answer_index"]
    return f"The answer is ({option_list[idx]}) {example['options'][idx]}"


def _format_gpqa_query(example):
    return (
        f"{example['Question']}\n\n"
        "Choose the correct answer from the following options:\n"
        f"(A) {example['Correct Answer']}\n"
        f"(B) {example['Incorrect Answer 1']}\n"
        f"(C) {example['Incorrect Answer 2']}\n"
        f"(D) {example['Incorrect Answer 3']}"
    )


def _format_rows(dataset_name: str, dataset):
    if dataset_name == "MATH":
        return [
            {
                "query": example["problem"],
                "gt": example["answer"],
                "tag": [dataset_name, "math", example["subject"], f"Level {example['level']}"],
                "source": dataset_name,
            }
            for example in dataset
        ]

    if dataset_name == "GSM8K":
        return [
            {"query": example["question"], "gt": example["answer"], "tag": ["math"], "source": "GSM8K"}
            for example in dataset
        ]

    if dataset_name == "AQUA-RAT":
        return [
            {
                "query": _format_aqua_query(example),
                "gt": str(example["correct"]),
                "tag": ["math", "reasoning", "multiple-choice"],
                "source": "AQUA-RAT",
            }
            for example in dataset
        ]

    if dataset_name == "MMLU":
        return [
            {
                "query": _format_mmlu_query(example),
                "gt": _format_mmlu_gt(example),
                "tag": ["mmlu", example["subject"]],
                "source": "MMLU",
            }
            for example in dataset
        ]

    if dataset_name == "MMLU-Pro":
        return [
            {
                "query": _format_mmlu_pro_query(example),
                "gt": _format_mmlu_pro_gt(example),
                "tag": ["MMLU-Pro", example["category"], example["src"]],
                "source": "MMLU-Pro",
                "num_choices": len(example["options"]),
            }
            for example in dataset
        ]

    if dataset_name == "GSM-Hard":
        return [
            {
                "query": example["input"],
                "gt": str(example["target"]),
                "tag": ["math", "GSM-Hard"],
                "source": "GSM-Hard",
            }
            for example in dataset
        ]

    if dataset_name == "GPQA":
        return [
            {
                "query": _format_gpqa_query(example),
                "gt": f"(A) {example['Correct Answer']}",
                "tag": ["GPQA", example["High-level domain"], example["Subdomain"], example["Writer's Difficulty Estimate"]],
                "source": "GPQA",
            }
            for example in dataset
        ]

    if dataset_name == "AIME-2024":
        return [
            {
                "query": example["Problem"],
                "gt": example["Answer"],
                "tag": [dataset_name, "math"],
                "source": dataset_name,
            }
            for example in dataset
        ]

    raise ValueError(f"Unsupported dataset: {dataset_name}")


def _deduplicate_by_query(rows):
    seen = set()
    out = []
    for row in rows:
        query = row.get("query", "")
        if query and query not in seen:
            out.append(row)
            seen.add(query)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=None, help="Subset of dataset names")
    args = parser.parse_args()

    target = args.only if args.only else ALL_DATASETS
    for dataset_name in target:
        if dataset_name not in ALL_DATASETS:
            print(f"[skip] Unknown dataset: {dataset_name}")
            continue
        try:
            dataset, used = _load_for_dataset(dataset_name)
            rows = _format_rows(dataset_name, dataset)
            rows = [r for r in rows if str(r.get("query", "")).strip() and str(r.get("gt", "")).strip()]
            rows = _deduplicate_by_query(rows)

            out_path = OUT_DIR / f"{dataset_name}_val.json"
            with open(out_path, "w") as f:
                json.dump(rows, f, indent=2)
            print(f"[ok] {dataset_name}: {len(rows)} samples -> {out_path} (from {used})")
        except Exception as exc:
            print(f"[fail] {dataset_name}: {exc}")


if __name__ == "__main__":
    main()
