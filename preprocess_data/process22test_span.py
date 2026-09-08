import json
import os

path = "/anonymous/path/QE/data/wmt-qe-2022/test_data-gold_labels/task1_word-level"
for dir in os.listdir(path):
    srcs_path = f"/anonymous/path/QE/data/wmt-qe-2022/test_data-gold_labels/task1_word-level/{dir}/test.2022.src"
    mt_path = f"/anonymous/path/QE/data/wmt-qe-2022/test_data-gold_labels/task1_word-level/{dir}/test.2022.mt"
    mt_tok_path = f"/anonymous/path/QE/data/wmt-qe-2022/test_data-gold_labels/task1_word-level/{dir}/test.2022.mt_tok"
    tag_path = f"/anonymous/path/QE/data/wmt-qe-2022/test_data-gold_labels/task1_word-level/{dir}/test.2022.{dir}.tags"

    srcs = open(srcs_path, 'r', encoding='utf-8')
    mts = open(mt_path, 'r', encoding='utf-8')
    mt_toks = open(mt_tok_path, 'r', encoding='utf-8')
    tags = open(tag_path, 'r', encoding='utf-8')

    results = []
    while True:
        src = srcs.readline().strip()
        mt = mts.readline().strip()
        mt_tok = mt_toks.readline().strip()
        tag = tags.readline().strip()

        if not src:
            break

        results.append({
            "src": src,
            "mt": mt,
            "tags": tag,
            "mt_tok": mt_tok
        })

    
    with open(f"/anonymous/path/QE/data/processed_span/{dir}-2022-test-span.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)