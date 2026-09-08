import json
import random

random.seed(42)
lang_pairs_span = [
    "en-de",
    # "en-es",
    "en-ru",
    # "en-zh",
    # "ja-zh",
    "zh-en",
]
lang_pairs_word = [
    "en-de",
    # "en-mr",
    # "en-zh",
    # "et-en",
    # "ne-en",
    # "ro-en",
    # "ru-en",
    # "si-en",
]
output_path = f"/anonymous/path/LLaMA-Factory/data/subtask_nonCOT-unit-tgt-3mixed-new-scale.json"
all_results = []


for lang_pair in lang_pairs_span:
    paths = [
        # f"/anonymous/path/QE/subtasks/semanticUnit/data/semanticUnit_LF_10k-{lang_pair}.json",
        # f"/anonymous/path/QE/subtasks/errorDetection/data/sequentialErrorDetection_LF_nonCOT_10k-{lang_pair}.json",
        # f"/anonymous/path/QE/subtasks/errorSeverity/data/errorSeverity_LF_nonCOT_10k-{lang_pair}.json",
        # f"/anonymous/path/QE/subtasks/errorDetection/data/unitErrorDetection_LF_nonCOT_10k-{lang_pair}.json",
        # f"/anonymous/path/QE/subtasks/errorSeverity/data/unitErrorSeverity_LF_nonCOT_10k-{lang_pair}.json",

        f"/anonymous/path/QE/subtasks/semanticUnit/data/semanticUnit_LF_scale-{lang_pair}.json",
        f"/anonymous/path/QE/subtasks/errorDetection/data/sequentialErrorDetection_LF_nonCOT_scale-{lang_pair}.json",
        f"/anonymous/path/QE/subtasks/errorSeverity/data/errorSeverity_LF_nonCOT_scale-{lang_pair}.json",
        f"/anonymous/path/QE/subtasks/errorDetection/data/unitErrorDetection_LF_nonCOT_scale-{lang_pair}.json",
        f"/anonymous/path/QE/subtasks/errorSeverity/data/unitErrorSeverity_LF_nonCOT_scale-{lang_pair}.json",
    ]

    for path in paths:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(len(data), path)
        all_results.extend(data)


for lang_pair in lang_pairs_word:
    paths = [
        f"/anonymous/path/QE/subtasks/semanticUnit/data/semanticUnit_LF_scale-{lang_pair}-word.json",
        f"/anonymous/path/QE/subtasks/errorDetection/data/sequentialErrorDetection_LF_nonCOT_scale-{lang_pair}-word.json",
        f"/anonymous/path/QE/subtasks/errorDetection/data/unitErrorDetection_LF_nonCOT_scale-{lang_pair}-word.json",
    ]

    for path in paths:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # if "semantic" in path and "mr" in path:
        #     data = random.sample(data, len(data) // 2)

        print(len(data), path)
        all_results.extend(data)


random.shuffle(all_results)
print(len(all_results))
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(all_results, f, ensure_ascii=False, indent=4)

