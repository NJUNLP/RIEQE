

import json
import random
from tqdm import tqdm

# random.seed(42)

# ==================================================
# 🔧 CONFIG
# ==================================================
lang_pair = "en-mr"
BASE_DIR = "/anonymous/path/QE/data/processed_train/w_span/splittedUnits"
BASE_DIR = "/anonymous/path/QE/data/processed_train/word_level/splittedUnits"

ERROR_PATH = f"{BASE_DIR}/{lang_pair}-Error.json"
NOERROR_PATH = f"{BASE_DIR}/{lang_pair}-noError.json"

output_path = f"data/sequentialErrorDetection_LF_nonCOT_scale-{lang_pair}-word.json"


prompt = """Given the following translation pair:

Source Sentence:
{src}
Target Sentence:
{tgt}

If we segment the target sentence into several semantic units:
{units}

Inspect each semantic unit **in order from left to right**.  
For each unit, check the following dimensions (no re-evaluation or backtracking):

1. **Terminology consistency** — Are key terms translated consistently and correctly?  
2. **Faithfulness / Accuracy** — Does the meaning match the source sentence precisely?  
   - Detect any mistranslation, omission, or addition.  
3. **Language naturalness and style** — Is the expression idiomatic, fluent, and appropriate in register?

For each semantic unit containing an error, identify the error span within it's corresponding translation.
- The `span` must be an **exact substring** from the target sentence.  
- The span should cover the **entire erroneous phrase** that conveys the incorrect meaning, not just an individual incorrect token.  
  - Include all words that together form the wrong meaning.  
  - Do **not** include unrelated correct words.  
- If there are multiple errors, list them **in the order they appear** in the target sentence.

Output in JSON format:
{{
    "Unit1": [
        "<error span 1>",
        "<error span 2>",
        ...
    ],
    ...
}}"""

# ==================================================
# 工具函数
# ==================================================
def find_start_index(str1: str, str2: str):
    idx = str2.lower().find(str1.lower())
    return idx if idx != -1 else None


# ==================================================
# 读取数据
# ==================================================
with open(ERROR_PATH, "r", encoding="utf-8") as f:
    error_data = json.load(f)

with open(NOERROR_PATH, "r", encoding="utf-8") as f:
    noerror_data = json.load(f)

random.seed(42)
random.shuffle(error_data)
random.seed(42)
random.shuffle(noerror_data)

error_data = error_data[1000:]
noerror_data = noerror_data[1000:]
# noerror_data = random.sample(noerror_data, len(noerror_data) // 3)
print("Error", len(error_data))
print("NoError", len(noerror_data))
random.seed(42)
noerror_data = random.sample(noerror_data, min(len(noerror_data), len(error_data)))

data = error_data + noerror_data
random.shuffle(data)
print(len(data))

# ==================================================
# 主处理逻辑
# ==================================================
results = []

for d in tqdm(data):
    src = d[f"src_{lang_pair.split('-')[0]}"]
    tgt = d[f"tgt_{lang_pair.split('-')[1]}"]
    units = d["Units"]
    errors = d.get("errors", [])

    instruction = prompt.format(
        src=src,
        tgt=tgt,
        units=json.dumps(units, ensure_ascii=False)
    )

    # ---------- locate unit spans ----------
    units_with_span = {}
    for uid, text in units.items():
        s = find_start_index(text, tgt)
        if s is None:
            continue
        units_with_span[uid] = {
            "text": text,
            "start": s,
            "end": s + len(text),
            "errors": []
        }

    if not units_with_span:
        continue

    # ---------- assign errors by CHAR-LEVEL INTERSECTION ----------
    for err in errors:
        es, ee = map(int, err["span"].split("-"))

        for unit in units_with_span.values():
            us, ue = unit["start"], unit["end"]

            inter_s = max(es, us)
            inter_e = min(ee, ue)

            if inter_s < inter_e:
                span_text = tgt[inter_s:inter_e]
                unit["errors"].append((inter_s, span_text))

    # ---------- build answer (ALL units) ----------
    answer = {}
    for uid, unit in units_with_span.items():
        if unit["errors"]:
            ordered_spans = [
                span for _, span in sorted(unit["errors"], key=lambda x: x[0])
            ]
            answer[uid] = ordered_spans
        else:
            answer[uid] = []

    results.append({
        "instruction": instruction,
        "answer": json.dumps(answer, ensure_ascii=False)
    })

# ==================================================
# 保存
# ==================================================
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print(f"\n✅ Saved {len(results)} samples to {output_path}")
print("\nExample:")
print(results[0]["instruction"])
print(results[0]["answer"])
print(results[1]["instruction"])
print(results[1]["answer"])