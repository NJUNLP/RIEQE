import json
import random
import unicodedata
from tqdm import tqdm

random.seed(42)

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

And we only examine the error span:
{span}

For this span, assign an error severity MINOR or MAJOR:
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


# ==================================================
# Config & Load data（对齐第一份代码）
# ==================================================

lang_pair = "ja-zh"
BASE_DIR = "/anonymous/path/QE/data/processed_train/w_span/splittedUnits"

ERROR_PATH = f"{BASE_DIR}/{lang_pair}-Error.json"

OUTPUT_PATH = f"data/unitErrorSeverity_LF_nonCOT_marker-{lang_pair}.json"

with open(ERROR_PATH, "r", encoding="utf-8") as f:
    error_samples = json.load(f)

random.seed(42)
random.shuffle(error_samples)
# error_samples = error_samples[1000:]
print(f"Using {len(error_samples)} error sentences")

# ==================================================
# Main loop
# ==================================================

results = []

for item in tqdm(error_samples):
    src = item[f"src_{lang_pair.split('-')[0]}"]
    tgt = item[f"tgt_{lang_pair.split('-')[1]}"]
    units = item["Units"]
    errors = item.get("errors", [])

    # ---------- filter GT errors ----------
    gt_errors = []
    for e in errors:
        es, ee = map(int, e["span"].split("-"))
        span_text = tgt[es:ee]
        if not is_only_punctuation(span_text):
            gt_errors.append(e)

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

    # ---------- collect ALL intersected error spans ----------
    all_spans = []              # 所有 error span text（去重前）
    span_records = []           # [(span_text, severity)]

    for err in gt_errors:
        es, ee = map(int, err["span"].split("-"))

        for us, ue in unit_spans.values():
            inter_s = max(es, us)
            inter_e = min(ee, ue)

            if inter_s < inter_e:
                span_text = tgt[inter_s:inter_e]

                if is_only_punctuation(span_text):
                    continue

                all_spans.append(span_text)
                span_records.append(
                    (span_text, err["type"])
                )

    if not span_records:
        continue

    # 去重但保持顺序
    all_spans = list(dict.fromkeys(all_spans))

    # ---------- one sample per focus span ----------
    for span_text, severity in span_records:

        instruction = prompt.format(
            src=src,
            tgt=tgt,
            units=json.dumps(units, ensure_ascii=False),
            spans=json.dumps(all_spans, ensure_ascii=False),
            span=span_text
        )

        answer = {
            "errors": [
                {
                    "span": span_text,
                    "severity": severity.upper()
                }
            ]
        }

        results.append({
            "instruction": instruction,
            "answer": json.dumps(answer, ensure_ascii=False)
        })

# ==================================================
# Save
# ==================================================

print(f"Total generated samples: {len(results)}")

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print("✅ DONE")
print(results[1]["instruction"])
print(results[1]["answer"])

print(len(results))