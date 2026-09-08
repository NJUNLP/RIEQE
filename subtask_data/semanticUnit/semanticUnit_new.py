import json
import random

random.seed(42)

# ==================================================
# Config
# ==================================================

lang_pair = "en-mr"
BASE_DIR = "/anonymous/path/QE/data/processed_train/w_span/splittedUnits"
BASE_DIR = "/anonymous/path/QE/data/processed_train/word_level/splittedUnits"

ERROR_PATH = f"{BASE_DIR}/{lang_pair}-Error.json"
NOERROR_PATH = f"{BASE_DIR}/{lang_pair}-noError.json"

TOTAL_NUM = 20000
ERROR_RATIO = 0.5
ERROR_NUM = int(TOTAL_NUM * ERROR_RATIO)
NOERROR_NUM = TOTAL_NUM - ERROR_NUM

ERROR_NUM = 10000
NOERROR_NUM = 8000

OUTPUT_PATH = f"data/semanticUnit_LF_10k-{lang_pair}.json"

# ==================================================
# Prompt
# ==================================================

prompt = (
    "Divide the sentence into **small semantic units** (around 5 words each).\n"
    "Each unit should represent a relatively complete piece of meaning.\n"
    "All units must be exact spans from the original sentence, and when concatenated in order, they should reconstruct the original sentence without any modification.\n"
    "Output in JSON format:\n"
    "{{\"Unit1\": <SemanticUnit1>, \"Unit2\": <SemanticUnit2>, ...}}\n\n"
    "Sentence:\n{tgt}"
)

# ==================================================
# Load data
# ==================================================

with open(ERROR_PATH, "r", encoding="utf-8") as f:
    error_data = json.load(f)

with open(NOERROR_PATH, "r", encoding="utf-8") as f:
    noerror_data = json.load(f)

assert len(error_data) >= ERROR_NUM
assert len(noerror_data) >= NOERROR_NUM

data = error_data[:ERROR_NUM] + noerror_data[:NOERROR_NUM]
random.shuffle(data)

print(f"Loaded {ERROR_NUM} error + {NOERROR_NUM} no-error samples")

# ==================================================
# Build training samples
# ==================================================

results = []

for item in data:
    tgt = item[f"tgt_{lang_pair.split('-')[1]}"]
    units = item["Units"]

    instruction = prompt.format(tgt=tgt)

    results.append({
        "instruction": instruction,
        "answer": json.dumps(units, ensure_ascii=False)
    })

# ==================================================
# Save
# ==================================================

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print(f"✅ Saved {len(results)} samples to {OUTPUT_PATH}")
print("\nExample:")
print(results[0]["instruction"])
print(results[0]["answer"])
print(len(results))