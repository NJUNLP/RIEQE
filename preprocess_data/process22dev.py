import pandas as pd
import json

file_path = '/anonymous/path/QE/data/wmt-qe-2022/train-dev_data/task1_mqm/dev/zh-en/zh-en-mqm.2022_dev.csv'
output_path = "/anonymous/path/QE/data/processed/zh-en-2022-dev.json" 
df = pd.read_csv(file_path, sep=',', header=0)
filtered_data = df[['src', 'mt', 'score', 'z_score']]
json_result = filtered_data.to_dict(orient='records')
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(json_result, f, indent=2, ensure_ascii=False)