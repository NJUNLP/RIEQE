import json
import random
import unicodedata
from tqdm import tqdm
import re

random.seed(42)

# ==================================================
# Utils
# ==================================================

def is_only_punctuation(s: str) -> bool:
    if not s:
        return False
    s = s.strip()
    if not s:
        return False
    return all(
        unicodedata.category(ch).startswith("P")
        for ch in s
        if not ch.isspace()
    )

def find_start(text: str, tgt: str):
    idx = tgt.lower().find(text.lower())
    return idx if idx != -1 else None


def split_sentence_to_units(sentence: str) -> str:
    """
    将句子按标点切分为 Unit1, Unit2...
    并避免:
    - 小数 (2.0, 3.14)
    - 千分位 (1,000)
    """

    # 切分规则
    pattern = r"""
        (?<!\d)[。！？!?]             # 句末标点（前面不是数字）
        | (?<!\d),(?!\d{3}\b)        # 逗号，但不是千分位
        | (?<!\d)\.(?!\d)            # 点号，但不是小数
        | [;；:：—…]+                # 其他分隔符
    """

    parts = re.split(pattern, sentence, flags=re.VERBOSE)

    # 清理空白
    parts = [p.strip() for p in parts if p.strip()]

    # 构造 Unit 结构
    units = {f"Unit{i+1}": part for i, part in enumerate(parts)}

    return units


# ==================================================
# Prompt
# ==================================================

prompt = """Given the following translation pair:

Source Sentence:
{src}
Target Sentence:
{tgt}

If we segment the target sentence into several semantic units:
{units}

And we know all the error spans:
{spans}

Assign each error span with an error severity MINOR or MAJOR:
- **MINOR** — Minor grammatical, stylistic, or fluency issues that do not significantly alter meaning.  
- **MAJOR** — Errors that change, distort, or obscure the meaning, including mistranslation, omission, or addition.

Output in JSON format:
{{
    "errors": [
        {{
            "span": "<error span>",
            "severity": "<MINOR or MAJOR>"
        }},
        ...
    ]
}}"""

# ==================================================
# Config（对齐第一份代码）
# ==================================================

lang_pair = "zh-en"
BASE_DIR = "/anonymous/path/QE/data/processed_train/w_span/splittedUnits"

ERROR_PATH = f"{BASE_DIR}/{lang_pair}-Error.json"

ERROR_SENT_NUM = 10000
OUTPUT_PATH = f"data/errorSeverity_LF_nonCOT_marker-{lang_pair}.json"

# ==================================================
# Load Error-only data
# ==================================================

with open(ERROR_PATH, "r", encoding="utf-8") as f:
    error_samples = json.load(f)

random.seed(42)
random.shuffle(error_samples)
# error_samples = error_data[1000:]

print(f"Using {len(error_samples)} error samples")

# ==================================================
# Main loop
# ==================================================

results = []

for item in tqdm(error_samples):
    src = item[f"src_{lang_pair.split('-')[0]}"]
    tgt = item[f"tgt_{lang_pair.split('-')[1]}"]
    # units = item["Units"]      # 已经切好的 units
    units = split_sentence_to_units(tgt)
    errors = item.get("errors", [])

    # ---------- filter GT errors ----------
    gt_errors = [
        e for e in errors
        if not is_only_punctuation(tgt[int(e["span"].split("-")[0]):int(e["span"].split("-")[1])])
    ]
    if not gt_errors:
        continue

    # ---------- locate unit spans ----------
    unit_spans = {}
    valid = True
    for uid, text in units.items():
        s = find_start(text, tgt)
        if s is None:
            valid = False
            break
        unit_spans[uid] = (s, s + len(text))
    if not valid:
        continue

    # ---------- assign errors ----------
    spans_input = []
    unit_answer = {"errors": []}

    for err in gt_errors:
        es, ee = map(int, err["span"].split("-"))

        for us, ue in unit_spans.values():
            inter_s = max(es, us)
            inter_e = min(ee, ue)

            if inter_s < inter_e:
                inter_text = tgt[inter_s:inter_e]

                if is_only_punctuation(inter_text):
                    continue

                spans_input.append(inter_text)
                unit_answer["errors"].append({
                    "span": inter_text,
                    "severity": err["type"].upper()
                })

    if not unit_answer["errors"]:
        continue

    # ---------- build instruction ----------
    instruction = prompt.format(
        src=src,
        tgt=tgt,
        units=json.dumps(units, ensure_ascii=False),
        spans=spans_input
    )

    results.append({
        "instruction": instruction,
        "answer": json.dumps(unit_answer, ensure_ascii=False)
    })

# ==================================================
# Save
# ==================================================

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print("✅ DONE")
print(f"Final samples: {len(results)}")
print(results[0]["instruction"])
print(results[0]["answer"])

print(results[1]["instruction"])
print(results[1]["answer"])
print(len(results))