#!/usr/bin/env python3
"""
数据预处理脚本：加载、清洗和预处理双色球历史数据
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

def load_data(csv_path: str) -> pd.DataFrame:
    """加载CSV数据"""
    df = pd.read_csv(csv_path)
    print(f"加载数据：{len(df)} 行，{len(df.columns)} 列")
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """数据清洗"""
    # 删除缺失值
    df = df.dropna()
    
    # 转换日期格式
    df['date'] = pd.to_datetime(df['date'].str[:10])
    
    # 按日期排序
    df = df.sort_values('date').reset_index(drop=True)
    
    print(f"清洗后数据：{len(df)} 行")
    return df

def create_sequences(data: np.ndarray, seq_length: int = 10):
    """创建时间序列数据"""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

def prepare_features(df: pd.DataFrame) -> tuple:
    """准备特征和目标"""
    # 选择特征列
    feature_columns = [
        'red1', 'red2', 'red3', 'red4', 'red5', 'red6', 'blue',
        'red_sum', 'red_range', 'red_avg', 'red_std',
        'odd_count', 'even_count', 'small_count', 'large_count',
        'zone1_count', 'zone2_count', 'zone3_count'
    ]
    
    # 添加频率特征（可选）
    freq_columns = [col for col in df.columns if 'freq' in col or 'omi' in col]
    feature_columns.extend(freq_columns)
    
    # 提取特征
    features = df[feature_columns].values.astype(np.float32)
    
    # 归一化
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)
    
    # 保存scaler参数供后续使用
    scaler_params = {
        'min': scaler.data_min_.tolist(),
        'max': scaler.data_max_.tolist()
    }
    
    return features_scaled, scaler, scaler_params

def main():
    # 路径配置
    base_dir = Path(__file__).parent.parent
    csv_path = base_dir / 'ssq_history.csv'
    output_dir = base_dir / 'data'
    
    # 加载和清洗数据
    df = load_data(csv_path)
    df = clean_data(df)
    
    # 准备特征
    features_scaled, scaler, scaler_params = prepare_features(df)
    
    # 创建序列数据
    seq_length = 10
    X, y = create_sequences(features_scaled, seq_length)
    
    # 划分训练集和测试集
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # 保存预处理数据
    np.save(output_dir / 'X_train.npy', X_train)
    np.save(output_dir / 'X_test.npy', X_test)
    np.save(output_dir / 'y_train.npy', y_train)
    np.save(output_dir / 'y_test.npy', y_test)
    
    # 保存scaler参数
    with open(output_dir / 'scaler_params.json', 'w') as f:
        json.dump(scaler_params, f)
    
    # 保存特征列名
    feature_columns = [
        'red1', 'red2', 'red3', 'red4', 'red5', 'red6', 'blue',
        'red_sum', 'red_range', 'red_avg', 'red_std',
        'odd_count', 'even_count', 'small_count', 'large_count',
        'zone1_count', 'zone2_count', 'zone3_count'
    ]
    freq_columns = [col for col in df.columns if 'freq' in col or 'omi' in col]
    feature_columns.extend(freq_columns)
    
    with open(output_dir / 'feature_columns.json', 'w') as f:
        json.dump(feature_columns, f)
    
    print(f"预处理完成：")
    print(f"  训练集：{X_train.shape}")
    print(f"  测试集：{X_test.shape}")
    print(f"  序列长度：{seq_length}")
    print(f"  特征数量：{X_train.shape[2]}")

if __name__ == '__main__':
    main()
