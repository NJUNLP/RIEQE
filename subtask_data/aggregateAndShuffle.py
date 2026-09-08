import json
import random

random.seed(42)
lang_pair = "zh-en"
paths = [
    f"/anonymous/path/QE/subtasks/semanticUnit/data/semanticUnit_LF_10k-{lang_pair}.json",
    f"/anonymous/path/QE/subtasks/errorDetection/data/sequentialErrorDetection_LF_nonCOT_10k-{lang_pair}.json",
    f"/anonymous/path/QE/subtasks/errorSeverity/data/errorSeverity_LF_nonCOT_10k-{lang_pair}.json",

    f"/anonymous/path/QE/subtasks/errorDetection/data/unitErrorDetection_LF_nonCOT_10k-{lang_pair}.json",
    f"/anonymous/path/QE/subtasks/errorSeverity/data/unitErrorSeverity_LF_nonCOT_10k-{lang_pair}.json",
]


output_path = f"/anonymous/path/LLaMA-Factory/data/subtask_nonCOT-unit-tgt-{lang_pair}-new.json"

all_results = []
for path in paths:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    all_results.extend(data)

random.shuffle(all_results)
print(len(all_results))
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(all_results, f, ensure_ascii=False, indent=4)

