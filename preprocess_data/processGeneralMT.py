import csv
import json
import os
import math

import re

def extract_text_and_v_content_with_span(s):
    # 查找 <v>...</v> 的位置和内容
    match = re.search(r'<v>(.*?)</v>', s)
    
    if not match:
        return s, "", -1, -1  # 没找到则返回无效位置
    
    v_content = match.group(1)
    start_v_tag = match.start()   # '<v>' 的起始位置
    end_v_tag = match.end()       # '</v>' 的结束位置
    len_v_open = len('<v>')       # '<v>' 的长度 (3)
    len_v_close = len('</v>')     # '</v>' 的长度 (4)

    # 构建 cleaned_string：移除 <v> 和 </v>
    cleaned_string = s[:start_v_tag] + s[start_v_tag + len_v_open : end_v_tag - len_v_close] + s[end_v_tag:]

    # 计算 v_content 在 cleaned_string 中的起始位置
    # 在 cleaned_string 中，v_content 就是从 start_v_tag 开始的（因为前面没变，只是少了 '<v>'）
    start_in_cleaned = start_v_tag
    end_in_cleaned = start_in_cleaned + len(v_content)

    # 验证
    assert cleaned_string[start_in_cleaned:end_in_cleaned] == v_content, "Span error!"

    return cleaned_string, v_content, start_in_cleaned, end_in_cleaned


# configs
src_l = "ja"
tgt_l = "zh"
MINOR = 100
MAJOR = 500

# 文件路径
inputPath = f"/anonymous/path/QE/data/generalMT/generalMT2024/mqm_generalMT2024_{src_l}{tgt_l}.tsv"
output_json_path = f"/anonymous/path/QE/data/processed_train/w_span/raw/{src_l}-{tgt_l}-generalMT24.json"

samples = [{
    "info": "",
    "src": "",
    "tgt": "",
    "error_span": [],
    "span": [],
    "severity": [],
}]
with open(inputPath, 'r', encoding='utf-8') as file:
    for row in file:
        if row.startswith("system	doc"):
            continue
        row = row.strip().split("\t")
        info = " ".join(row[:5])
        src = row[5]
        tgt = row[6]
        sev = row[8]
        tgt, err, start, end = extract_text_and_v_content_with_span(tgt)
        if info == samples[-1]['info'] and src == samples[-1]['src'] and tgt == samples[-1]["tgt"]:
            if err != "":
                samples[-1]['error_span'].append(err)
                samples[-1]['severity'].append(sev)
                samples[-1]['span'].append((start, end))
        else:
            samples.append({
                "info": info,
                "src": src,
                "tgt": tgt,
                "error_span": [err],
                "span": [(start, end)],
                "severity": [sev],
            })
    
samples = samples[1:]
print(len(samples))

results = []
scores = []

for item in samples:
    errors = []
    score = 100.0
    src = item['src']
    tgt = item['tgt']
    # 归一化权重：1 / (M * log2(N * len + C))
    M = 5
    N = 1
    C = 1
    tgt_word_count = len(tgt.split(' '))
    length_weight = 1 / (M * math.log(N * tgt_word_count + C, 2))

    for i in range(len(item['error_span'])):
        if item['error_span'][i] != "":  # 存在错误

            start_idx = item['span'][i][0]
            end_idx = item['span'][i][1]

            errors.append({
                'span': f"{start_idx}-{end_idx}",
                'type': item['severity'][i],
                'text': tgt[start_idx:end_idx]
            })
            penalty = MINOR if item['severity'][i].lower() == 'minor' else MAJOR
            score -= length_weight * penalty

    score = max(score, 0)
    scores.append(score)
    results.append({
        f"src_{src_l}": src.replace("<v>","").replace("</v>",""),
        f"tgt_{tgt_l}": tgt,
        "norm_score": score,
        "errors": errors
    })

# === 保存 JSON ===
os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
with open(output_json_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)