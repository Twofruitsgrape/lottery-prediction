#!/usr/bin/env python3
"""将历史数据转换为JSON格式供网页使用"""

import pandas as pd
import json
from pathlib import Path

# 读取CSV数据
csv_path = Path(__file__).parent.parent / 'ssq_history.csv'
df = pd.read_csv(csv_path)

# 转换为JSON格式
history_data = []
for _, row in df.iterrows():
    history_data.append({
        'issue': str(row['issue']),
        'date': str(row['date'])[:10],
        'red': [int(row['red1']), int(row['red2']), int(row['red3']), 
                int(row['red4']), int(row['red5']), int(row['red6'])],
        'blue': int(row['blue']),
        'red_sum': int(row['red_sum']),
        'red_range': int(row['red_range']),
        'odd_count': int(row['odd_count']),
        'even_count': int(row['even_count'])
    })

# 保存为JSON
output_path = Path(__file__).parent.parent / 'static' / 'history.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(history_data, f, ensure_ascii=False, indent=2)

print(f"已导出 {len(history_data)} 期历史数据到 {output_path}")
