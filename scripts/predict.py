#!/usr/bin/env python3
"""
预测脚本：加载模型并预测下一期号码
"""

import numpy as np
import tensorflow as tf
from pathlib import Path
import json
import pandas as pd

def load_model_and_data():
    """加载模型和数据"""
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data'
    model_dir = base_dir / 'model'
    
    # 加载模型（尝试新的Keras格式）
    model_path = model_dir / 'lottery_model.keras'
    if model_path.exists():
        model = tf.keras.models.load_model(model_path)
    else:
        # 回退到旧格式
        model_path = model_dir / 'lottery_model.h5'
        model = tf.keras.models.load_model(model_path)
    
    # 加载最新数据
    df = pd.read_csv(base_dir / 'ssq_history.csv')
    df = df.sort_values('issue').reset_index(drop=True)
    
    # 加载scaler参数
    with open(data_dir / 'scaler_params.json', 'r') as f:
        scaler_params = json.load(f)
    
    # 加载特征列名
    with open(data_dir / 'feature_columns.json', 'r') as f:
        feature_columns = json.load(f)
    
    return model, df, scaler_params, feature_columns

def preprocess_last_sequence(df, scaler_params, feature_columns, seq_length=10):
    """预处理最后10期数据"""
    # 提取特征
    features = df[feature_columns].values.astype(np.float32)
    
    # 归一化
    min_vals = np.array(scaler_params['min'])
    max_vals = np.array(scaler_params['max'])
    features_scaled = (features - min_vals) / (max_vals - min_vals)
    
    # 取最后10期
    last_sequence = features_scaled[-seq_length:]
    return last_sequence.reshape(1, seq_length, -1)

def predict_numbers(model, sequence, df):
    """预测号码"""
    # 预测（归一化值）
    prediction_scaled = model.predict(sequence, verbose=0)[0]
    
    # 生成红球（6个，不重复，1-33）
    red_probs = prediction_scaled[:33]  # 前33个是红球概率
    red_probs = red_probs / red_probs.sum()  # 归一化为概率
    red_numbers = np.random.choice(range(1, 34), size=6, replace=False, p=red_probs)
    red_numbers = np.sort(red_numbers)
    
    # 生成蓝球（1个，1-16）
    blue_probs = prediction_scaled[33:49]  # 33-48是蓝球概率
    blue_probs = blue_probs / blue_probs.sum()
    blue_number = np.random.choice(range(1, 17), p=blue_probs)
    
    return red_numbers.tolist(), int(blue_number)

def generate_prediction_output(red_numbers, blue_number, last_issue):
    """生成预测结果输出"""
    next_issue = str(int(last_issue) + 1)
    
    result = {
        'next_issue': next_issue,
        'red_numbers': red_numbers,
        'blue_number': blue_number,
        'timestamp': pd.Timestamp.now().isoformat(),
        'model_info': {
            'type': 'LSTM',
            'seq_length': 10,
            'features': 116
        }
    }
    
    return result

def main():
    # 加载模型和数据
    model, df, scaler_params, feature_columns = load_model_and_data()
    
    # 预处理最后序列
    sequence = preprocess_last_sequence(df, scaler_params, feature_columns)
    
    # 预测
    red_numbers, blue_number = predict_numbers(model, sequence, df)
    
    # 生成输出
    last_issue = df['issue'].iloc[-1]
    result = generate_prediction_output(red_numbers, blue_number, last_issue)
    
    # 保存预测结果
    output_dir = Path(__file__).parent.parent / 'static'
    output_path = output_dir / 'prediction.json'
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"预测完成：")
    print(f"  下一期期号：{result['next_issue']}")
    print(f"  红球预测：{result['red_numbers']}")
    print(f"  蓝球预测：{result['blue_number']}")
    print(f"  结果已保存到：{output_path}")

if __name__ == '__main__':
    main()
