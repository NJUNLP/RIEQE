import json
import os

datasets = [
    "en-de",
    "en-mr",
    "en-zh",
    "et-en",
    "ne-en",
    "ro-en",
    "ru-en",
    "si-en",
]

for dataset in datasets:
    mts_path = f"/anonymous/path/QE/data/wmt-qe-2022/train-dev_data/task1_word-level/train/{dataset}/{dataset}-train/train.mt"
    srcs_path = f"/anonymous/path/QE/data/wmt-qe-2022/train-dev_data/task1_word-level/train/{dataset}/{dataset}-train/train.src"
    mt_toks_path = f"/anonymous/path/QE/data/wmt-qe-2022/train-dev_data/task1_word-level/train/{dataset}/{dataset}-train/train.word_level.2022.mt"
    tags_path = f"/anonymous/path/QE/data/wmt-qe-2022/train-dev_data/task1_word-level/train/{dataset}/{dataset}-train/train.word_level.2022.tags"

    mts = open(mts_path, 'r', encoding='utf-8')
    srcs = open(srcs_path, 'r', encoding='utf-8')
    mt_toks = open(mt_toks_path, 'r', encoding='utf-8')
    tags = open(tags_path, 'r', encoding='utf-8')

    results = []
    while True:
        mt = mts.readline().strip()
        src = srcs.readline().strip()
        mt_tok = mt_toks.readline().strip()
        tag = tags.readline().strip()

        if not mt:
            break

        results.append({
            "src": src,
            "mt": mt,
            "tags": tag,
            "mt_tok": mt_tok,
        })

    print(dataset, len(results))
    with open(f"/anonymous/path/QE/data/processed_train/word_level/{dataset}-2022.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)