import os
import json

def construct_dict_from_lines(lines):
    keys = [
        "src", "mt", "score", "z_score"
    ]
    return {
        key: line.strip()
        for key, line in zip(keys, lines)
    }

def read_files_line_by_line(file_paths, construct_func=construct_dict_from_lines):
    if not file_paths:
        return []

    files = [open(fp, 'r', encoding='utf-8') for fp in file_paths]
    result = []

    while True:
        lines = []
        for f in files:
            line = f.readline()
            if not line:
                break
            lines.append(line)

        if len(lines) != len(files):
            break

        row_dict = construct_func(lines)
        result.append(row_dict)

    for f in files:
        f.close()

    return result

# 示例用法：
if __name__ == "__main__":
    files = [
        '/anonymous/path/QE/data/wmt-qe-2022/test_data-gold_labels/task1_mqm/zh-en/test.2022.src', 
        '/anonymous/path/QE/data/wmt-qe-2022/test_data-gold_labels/task1_mqm/zh-en/test.2022.mt', 
        '/anonymous/path/QE/data/wmt-qe-2022/test_data-gold_labels/task1_mqm/zh-en/test.2022.zh-en.mqm_score.mqm',
        '/anonymous/path/QE/data/wmt-qe-2022/test_data-gold_labels/task1_mqm/zh-en/test.2022.zh-en.mqm_z_score'
    ]
    data = read_files_line_by_line(files, construct_dict_from_lines)
    for d in data:
        d['score'] = float(d['score'])
        d['z_score'] = float(d['z_score'])
    with open('/anonymous/path/QE/data/processed/zh-en-2022-test.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)