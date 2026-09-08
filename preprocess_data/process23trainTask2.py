import csv
import json
import os
import math
import matplotlib.pyplot as plt

# configs
src_l = "en"
tgt_l = "ru"
MINOR = 100
MAJOR = 500

# 文件路径
inputPath = "/anonymous/path/QE/data/wmt-qe-2023/task_2/train/2021_TED_en-ru_processed.tsv"
# inputPath = "/anonymous/path/QE/data/wmt-qe-2023/task_2/train/2021_en-ru_processed.tsv"
# inputPath = "/anonymous/path/QE/data/wmt-qe-2023/task_2/train/2020_en-ru_processed.tsv"

# inputPath = "/anonymous/path/QE/data/wmt-qe-2023/task_2/train/2021_TED_en-de_processed.tsv"
# inputPath = "/anonymous/path/QE/data/wmt-qe-2023/task_2/train/2021_en-de_processed.tsv"
# inputPath = "/anonymous/path/QE/data/wmt-qe-2023/task_2/train/2020_en-de_processed.tsv"

output_json_path = "/anonymous/path/QE/data/processed_train/w_span/en-ru-21-ted.json"
output_plot_path = "/anonymous/path/QE/data/processed_train/distributions/en-ru-21-ted-score_dist.png"

results = []
scores = []        # norm_score
tgt_lengths = []   # 对应的目标句词数长度

with open(inputPath, 'r', encoding='utf-8') as file:
    for row in file:
        if row.startswith("mt_system	doc_id"):
            continue
        row = row.strip().split("\t")
        # if len(row) < 9:
        #     continue  # 跳过格式错误行
        src = row[4]
        tgt = row[5]
        starts = row[6].split(' ')
        ends = row[7].split(' ')
        sevs = row[8].split(' ')

        errors = []
        score = 100.0
        # 归一化权重：1 / (M * log2(N * len + C))
        M = 5
        N = 1
        C = 1
        tgt_word_count = len(tgt.split(' '))
        length_weight = 1 / (M * math.log(N * tgt_word_count + C, 2))

        for i in range(len(starts)):
            if starts[0] != '-1':  # 存在错误
                start_idx = int(starts[i])
                end_idx = int(ends[i])
                errors.append({
                    'span': f"{starts[i]}-{ends[i]}",
                    'type': sevs[i],
                    'text': tgt[start_idx:end_idx]
                })
                penalty = MINOR if sevs[i] == 'minor' else MAJOR
                score -= length_weight * penalty

        score = max(score, 0)
        scores.append(score)
        tgt_lengths.append(tgt_word_count)
        results.append({
            f"src_{src_l}": src,
            f"tgt_{tgt_l}": tgt,
            "norm_score": score,
            "errors": errors
        })

# === 保存 JSON ===
os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
with open(output_json_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

# === 绘制柱状图 + 每个 bin 的平均 tgt 长度 ===
plt.figure(figsize=(12, 6))
n_bins = 20
bin_edges = [i * 5 for i in range(n_bins + 1)]  # 0,5,10,...,100

# 使用 numpy digitize 分配每个 score 到 bin
import numpy as np
bin_indices = np.digitize(scores, bin_edges)  # 返回每个 score 所属 bin 的索引
bin_indices = np.clip(bin_indices, 1, n_bins)  # 映射到 1~20

# 按 bin 聚合平均长度
bin_avg_lengths = {}
for i in range(len(scores)):
    b = bin_indices[i]
    if b not in bin_avg_lengths:
        bin_avg_lengths[b] = []
    bin_avg_lengths[b].append(tgt_lengths[i])

# 计算每个 bin 的平均长度
bin_centers = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(n_bins)]
avg_lengths_per_bin = [
    np.mean(bin_avg_lengths.get(i+1, [0])) for i in range(n_bins)
]

# 绘制柱状图
n, bins_, patches = plt.hist(scores, bins=bin_edges, edgecolor='black', alpha=0.7, color='skyblue')

# 在每个 bar 上方标注平均长度
for i in range(n_bins):
    if n[i] > 0:  # 有数据才标注
        x_pos = (bin_edges[i] + bin_edges[i+1]) / 2
        y_pos = n[i] + 0.01 * max(n)  # 稍微高出 bar
        avg_len = avg_lengths_per_bin[i]
        plt.text(x_pos, y_pos, f"{avg_len:.1f}", ha='center', va='bottom', fontsize=9, color='darkred', weight='bold')

plt.title(f'Distribution of norm_score ({src_l}→{tgt_l}, N={len(scores)})', fontsize=14)
plt.xlabel('Normalized Score', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.xticks(range(0, 101, 10))
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 保存图像
os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)
plt.tight_layout()
plt.savefig(output_plot_path, dpi=200)
plt.close()

print(f"✅ JSON saved to: {output_json_path}")
print(f"📊 Histogram saved to: {output_plot_path}")
print(f"📈 Score stats: min={min(scores):.2f}, max={max(scores):.2f}, mean={sum(scores)/len(scores):.2f}")