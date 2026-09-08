import json
import random
from tqdm import tqdm

# random.seed(42)

# ==================================================
# 🔧 CONFIG
# ==================================================
lang_pair = "en-mr"
BASE_DIR = "/anonymous/path/QE/data/processed_train/w_span/splittedUnits"
# BASE_DIR = "/anonymous/path/QE/data/processed_train/word_level/splittedUnits"

ERROR_PATH = f"{BASE_DIR}/{lang_pair}-Error.json"
NOERROR_PATH = f"{BASE_DIR}/{lang_pair}-noError.json"

output_path = f"./all_task_LF_nonCOT_scale-{lang_pair}.json"


prompt = """You are doing a translation quality estimation task.

Given the following translation pair:

Source Sentence:
{src}
Target Sentence:
{tgt}

To find translation errors, check the following dimensions (no re-evaluation or backtracking):

1. **Terminology consistency** — Are key terms translated consistently and correctly?  
2. **Faithfulness / Accuracy** — Does the meaning match the source sentence precisely?  
   - Detect any mistranslation, omission, or addition.  
3. **Language naturalness and style** — Is the expression idiomatic, fluent, and appropriate in register?

For each error, identify the error span within it's corresponding translation.
- The `span` must be an **exact substring** from the target sentence.  
- The span should cover the **entire erroneous phrase** that conveys the incorrect meaning, not just an individual incorrect token.  
  - Include all words that together form the wrong meaning.  
  - Do **not** include unrelated correct words.  
- If there are multiple errors, list them **in the order they appear** in the target sentence.

For each error span, assign an error severity MINOR or MAJOR:
- **MINOR** — Minor grammatical, stylistic, or fluency issues that do not significantly alter meaning.  
- **MAJOR** — Errors that change, distort, or obscure the meaning, including mistranslation, omission, or addition.

Output in JSON format:
{{
    "errors": [
        {{
            "span": "<error span>",
            "severity": "<MINOR or MAJOR>"
        }}
    ]
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
    errors = d.get("errors", [])

    instruction = prompt.format(
        src=src,
        tgt=tgt,
    )

    # ---------- build answer (ALL units) ----------
    answer = {
        "errors": [
            {
                "span": item['text'],
                "severity": item['type']
            } for item in errors
        ]
    }
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