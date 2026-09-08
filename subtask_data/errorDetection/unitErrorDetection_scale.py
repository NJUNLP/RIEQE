import json
import random
from tqdm import tqdm

random.seed(42)

# ==================================================
# 🔧 CONFIG
# ==================================================
lang_pair = "en-mr"
BASE_DIR = "/anonymous/path/QE/data/processed_train/w_span/splittedUnits"
BASE_DIR = "/anonymous/path/QE/data/processed_train/word_level/splittedUnits"

ERROR_PATH = f"{BASE_DIR}/{lang_pair}-Error.json"
NOERROR_PATH = f"{BASE_DIR}/{lang_pair}-noError.json"

OUTPUT_PATH = f"data/unitErrorDetection_LF_nonCOT_scale-{lang_pair}-word.json"

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

And we only examine this semantic unit:
{unit}

For this unit, check the following dimensions (no re-evaluation or backtracking):

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
    "errors": [
        "<error span 1>",
        "<error span 2>",
        ...
    ]
}}"""

# ==================================================
# 工具函数
# ==================================================
def find_start_index(sub: str, text: str):
    idx = text.lower().find(sub.lower())
    return idx if idx != -1 else None


# ==================================================
# Load sentence-level data (NO RANDOM)
# ==================================================
with open(ERROR_PATH, "r", encoding="utf-8") as f:
    error_sentences = json.load(f)

with open(NOERROR_PATH, "r", encoding="utf-8") as f:
    noerror_sentences = json.load(f)

random.seed(42)
random.shuffle(error_sentences)
random.seed(42)
random.shuffle(noerror_sentences)

error_sentences = error_sentences[1000:]
noerror_sentences = noerror_sentences[1000:]
random.seed(42)
noerror_sentences = random.sample(noerror_sentences, min(len(noerror_sentences), len(error_sentences)))


print(f"Using {len(error_sentences)} ERROR sentences")
print(f"Using {len(noerror_sentences)} NO-ERROR sentences")

# ==================================================
# Unit-level pools
# ==================================================
error_units = []
noerror_units = []

def process_sentence(item):
    src = item[f"src_{lang_pair.split('-')[0]}"]
    tgt = item[f"tgt_{lang_pair.split('-')[1]}"]
    units = item["Units"]
    errors = item.get("errors", [])

    # ---- locate unit spans ----
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
        return

    # ---- assign errors (char-level intersection) ----
    for err in errors:
        es, ee = map(int, err["span"].split("-"))
        for unit in units_with_span.values():
            us, ue = unit["start"], unit["end"]
            inter_s = max(es, us)
            inter_e = min(ee, ue)
            if inter_s < inter_e:
                unit["errors"].append((inter_s, tgt[inter_s:inter_e]))

    # ---- build unit samples ----
    for unit in units_with_span.values():
        instruction = prompt.format(
            src=src,
            tgt=tgt,
            units=json.dumps(units, ensure_ascii=False),
            unit=unit["text"]
        )

        ordered_spans = [
            span for _, span in sorted(unit["errors"], key=lambda x: x[0])
        ]

        sample = {
            "instruction": instruction,
            "answer": json.dumps({"errors": ordered_spans}, ensure_ascii=False)
        }

        if ordered_spans:
            error_units.append(sample)
        else:
            noerror_units.append(sample)


# ==================================================
# Build unit pools
# ==================================================
for item in tqdm(error_sentences, desc="Processing ERROR sentences"):
    process_sentence(item)

for item in tqdm(noerror_sentences, desc="Processing NO-ERROR sentences"):
    process_sentence(item)

print(f"\nTotal error units: {len(error_units)}")
print(f"Total no-error units: {len(noerror_units)}")

# ==================================================
# 🔥 FINAL SAMPLING (KEY CHANGE)
# ==================================================
KEEP_NOERROR = int(1 * len(error_units))
KEEP_NOERROR = min(KEEP_NOERROR, len(noerror_units))

final_units = (
    error_units +
    random.sample(noerror_units, KEEP_NOERROR)
)

random.shuffle(final_units)

# ==================================================
# Save
# ==================================================
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(final_units, f, ensure_ascii=False, indent=4)

print("\n✅ DONE")
print(f"Error units: {len(error_units)}")
print(f"Final unit-level samples: {len(final_units)}")

# print(final_units[0]['instruction'])
# print(final_units[0]['answer'])
# print(final_units[1]['instruction'])
# print(final_units[1]['answer'])