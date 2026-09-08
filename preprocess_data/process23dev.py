import pandas as pd
import json

file_path = '/anonymous/path/QE/data/wmt-qe-2023/task_1/en-te/dev.ente.df.short.tsv'
output_path = "/anonymous/path/QE/data/processed/en-te-2023-dev.json" 
df = pd.read_csv(file_path, sep='\t', header=0)
df.rename(columns={
    'original': 'src',
    'translation': 'mt',
    'mean': 'score',
    'z_mean': 'z_score'
}, inplace=True)

json_result = df[['src', 'mt', 'score', 'z_score']].to_dict(orient='records')

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(json_result, f, indent=2, ensure_ascii=False)