import csv
import json
import matplotlib.pyplot as plt
from collections import Counter

SCORE_DIFF_THRESHOLD = 50

src_l = 'en'
tgt_l = 'gu'

path = f"/anonymous/path/QE/data/wmt-qe-2023/task_1/{src_l}-{tgt_l}/train.{src_l}{tgt_l}.df.short.tsv"

results = []
cnt = 0
with open(path, 'r', encoding='utf-8') as f:
    for row in f:
        cnt += 1
        try:
            row = row.split("\t")
            src = row[1]
            tgt = row[2]
            scores = eval(row[3].replace("\"", ""))
            score = row[4]
            
            if max(scores) - min(scores) >= SCORE_DIFF_THRESHOLD:
                continue

            results.append({
                f'src_{src_l}': src,
                f'tgt_{tgt_l}': tgt,
                "norm_score": float(score),
                "errors": []
            })
        except Exception as e:
            print(e)
            continue

print(cnt)
print(len(results))
with open(f"/anonymous/path/QE/data/processed_train/wo_span/{src_l}-{tgt_l}.json", 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

scores = [d['norm_score'] for d in results]
# 统计每个分数出现的次数
import numpy as np

# 定义分箱边界：0, 5, 10, ..., 100（共 21 个边界，20 个区间）
bins = np.arange(0, 105, 5)  # [0, 5, 10, ..., 100] → 21 个点，20 个区间
labels = [f"{i}-{i+5}" for i in range(0, 100, 5)]  # 标签：["0-5", "5-10", ..., "95-100"]

# 统计每个区间的频次
hist, _ = np.histogram(scores, bins=bins)

# 绘制柱状图
plt.figure(figsize=(14, 6))
bars = plt.bar(labels, hist, color='lightcoral', edgecolor='black', width=0.8)

# 标题和标签
plt.title('Score Distribution (0-100, 20 Bins)', fontsize=16)
plt.xlabel('Score Range', fontsize=12)
plt.ylabel('Count', fontsize=12)

# 旋转 x 轴标签，避免重叠
plt.xticks(rotation=45, ha='right')

# 在每个柱子上方显示频次
for bar, count in zip(bars, hist):
    if count > 0:  # 只在有数据的柱子上显示
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 str(int(count)), ha='center', va='bottom', fontsize=9)

# 添加网格（y轴方向）
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 自动调整布局
plt.tight_layout()

# 保存图像
plt.savefig(f"/anonymous/path/QE/data/processed_train/distributions/{src_l}-{tgt_l}.png")
plt.close()