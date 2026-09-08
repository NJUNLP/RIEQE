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

OUTPUT_PATH = f"data/semanticUnit_LF_scale-{lang_pair}-word.json"

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