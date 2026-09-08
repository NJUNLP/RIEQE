import json
from tqdm import tqdm

lang_pair = "en-de"
input_path = f"/anonymous/path/QE/subtasks/errorDetection/data/unitSequentialErrorDetection_10_qwen3_14b_filtered-{lang_pair}.jsonl"
output_path = f"/anonymous/path/QE/subtasks/errorDetection/data/unitErrorDetection_LF_COT-{lang_pair}.json"

results = []
with open(input_path, 'r', encoding='utf-8') as fin:
    for line in tqdm(fin, desc="Processing"):
        line = line.strip()
        data = json.loads(line)
        if data.get("state", "") == "WRONG":
            continue
        temp = {
            "instruction": data['input'].split("<|im_start|>user")[-1].split("<|im_end|>")[0].strip(),
            "answer": data['generated'],
        }
        results.append(temp)

print(len(results))
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)