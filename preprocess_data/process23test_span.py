import os 
import json

t1_labels = "/anonymous/path/QE/data/wmt-qe-2023/gold_labels/hallucinations_gold_T1w.tsv"
t2_labels = "/anonymous/path/QE/data/wmt-qe-2023/gold_labels/hallucinations_gold_T2.tsv"
z_scores = "/anonymous/path/QE/data/wmt-qe-2023/gold_labels/hallucinations_gold_T1s.tsv"
labels1 = open(t1_labels, 'r', encoding='utf-8')
labels2 = open(t2_labels, 'r', encoding='utf-8')
scores = open(z_scores, 'r', encoding='utf-8')

label1 = ""
while not label1.startswith("en-de	gold	1896"):
    label1 = labels1.readline().strip()
# label2 = ""
# while not label2.startswith("zh-en	gold	1676"):
#     label2 = labels2.readline().strip()
score = ""
while not score.startswith("en-de	gold	1896"):
    score = scores.readline().strip()


path_src = "/anonymous/path/QE/data/wmt-qe-2023/test_data_2023/task1_word_level/en-mr/test.enmr.final.src"
srcs = open(path_src, 'r', encoding='utf-8')

results = []
while True:
    label1 = labels1.readline().strip()
    # label2 = labels2.readline().strip()
    src = srcs.readline().strip()
    score = scores.readline().strip().split("\t")[-1]

    if not label1 or not src:
        break
    if not label1.startswith("en-mr"):
        break

    if "hallucination" in label1:
        continue
    # sev = label2.split("\t")[-1]
    # ends = label2.split("\t")[-2]
    # starts = label2.split("\t")[-3]

    tags = label1.split("\t")[-1]
    mt_tok = label1.split("\t")[-2]
    if len(tags.split()) != len(mt_tok.split()):
        continue

    results.append({
        "src": src,
        "mt": label1.split("\t")[-3],
        "z_score": float(score),
        # "starts": [int(t) for t in starts.split(" ")],
        # "ends": [int(t) for t in ends.split(" ")],
        # "severities": sev.split(" "),
        "tags": tags,
        "mt_tok": mt_tok,
    })

print(len(results))
with open("/anonymous/path/QE/data/processed_span/en-mr-2023-test-span_original.json", 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)



