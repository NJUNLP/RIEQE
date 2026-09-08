import os 
import json

gold_labels = "/anonymous/path/QE/data/wmt-qe-2023/gold_labels/hallucinations_gold_T1s.tsv"
labels = open(gold_labels, 'r', encoding='utf-8')

file_dir = "/anonymous/path/QE/data/wmt-qe-2023/test_data_2023/task1_sentence_level"
while True:
    try:
        label = labels.readline()
        lang_pair = label.strip().split('\t')[0]
        mt_path = f"{file_dir}/{lang_pair}/test.{lang_pair.split('-')[0]}{lang_pair.split('-')[1]}.final.mt"
        src_path = f"{file_dir}/{lang_pair}/test.{lang_pair.split('-')[0]}{lang_pair.split('-')[1]}.final.src"
        mt = open(mt_path, 'r', encoding='utf-8')
        src = open(src_path, 'r', encoding='utf-8')
    except:
        break

    results = []
    lang_pair = None
    while True:
        
        mt_line = mt.readline()
        src_line = src.readline()
        if not mt_line:
            break
        if lang_pair is None:
            print("here")
            lang_pair = label.strip().split('\t')[0]
            l = label
        else:
            l = labels.readline()
        try:
            d = {
                'src': src_line.strip(),
                'mt': mt_line.strip(),
                'z_score': float(l.strip().split('\t')[-1])
            }
            results.append(d)
        except Exception as e:
            # print(e)
            continue

    with open(f"/anonymous/path/QE/data/processed/{lang_pair}-2023-test.json", 'w', encoding='utf-8') as f:
        print(lang_pair, len(results))
        json.dump(results, f, ensure_ascii=False, indent=4)