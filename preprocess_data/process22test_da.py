import json
import os

path = "/anonymous/path/QE/data/wmt-qe-2022/test_data-gold_labels/task1_da"
for dir in os.listdir(path):
    srcs_path = f"/anonymous/path/QE/data/wmt-qe-2022/test_data-gold_labels/task1_da/{dir}/test.2022.src"
    mt_path = f"/anonymous/path/QE/data/wmt-qe-2022/test_data-gold_labels/task1_da/{dir}/test.2022.mt"
    score_path = f"/anonymous/path/QE/data/wmt-qe-2022/test_data-gold_labels/task1_da/{dir}/test.2022.{dir}.da_score"

    srcs = open(srcs_path, 'r', encoding='utf-8')
    mts = open(mt_path, 'r', encoding='utf-8')
    scores = open(score_path, 'r', encoding='utf-8')

    results = []
    while True:
        src = srcs.readline().strip()
        mt = mts.readline().strip()
        score = scores.readline().strip()

        if not src:
            break

        results.append({
            "src": src,
            "mt": mt,
            "score": score,
        })

    
    with open(f"/anonymous/path/QE/data/processed/{dir}-2022-test.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)